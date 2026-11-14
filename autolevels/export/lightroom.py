"""Adobe Lightroom / Camera Raw curves-only develop presets."""
from pathlib import Path
from uuid import uuid4
import xml.etree.ElementTree as ET

import numpy as np

from .common import prepare_curves
from .darktable import fit_rgb_curves


CRS = 'http://ns.adobe.com/camera-raw-settings/1.0/'
RDF = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
XMP = 'adobe:ns:meta/'


def export_curves(path, curves):
    """Write an XMP preset with up to 16 integer control points per channel.

    The master curve is identity. Other image adjustments are omitted.
    Adobe interpolation and working space can change the rendered result.
    """
    path = Path(path)
    values = prepare_curves(curves)

    for prefix, uri in [('x', XMP), ('rdf', RDF), ('crs', CRS)]:
        ET.register_namespace(prefix, uri)

    root = ET.Element(f'{{{XMP}}}xmpmeta')
    rdf = ET.SubElement(root, f'{{{RDF}}}RDF')
    desc = ET.SubElement(rdf, f'{{{RDF}}}Description', {f'{{{RDF}}}about': ''})

    for key, value in {
        'PresetType': 'Normal', 'UUID': uuid4().hex.upper(),
        'SupportsAmount': 'False', 'SupportsColor': 'True',
        'SupportsMonochrome': 'True', 'SupportsNormalDynamicRange': 'True',
        'SupportsHighDynamicRange': 'False', 'SupportsSceneReferred': 'True',
        'SupportsOutputReferred': 'True', 'CameraModelRestriction': '',
        'HasSettings': 'True', 'ToneCurveName2012': 'Custom',
    }.items():
        desc.set(f'{{{CRS}}}{key}', value)

    for key, value in [('Name', f'{path.name}'), ('Group', 'AutoLevels')]:
        alt = ET.SubElement(ET.SubElement(desc, f'{{{CRS}}}{key}'), f'{{{RDF}}}Alt')
        ET.SubElement(alt, f'{{{RDF}}}li',
                      {'{http://www.w3.org/XML/1998/namespace}lang': 'x-default'}).text = value

    channels = [('ToneCurvePV2012', np.array([[0, 0], [255, 255]]))]
    splines = fit_rgb_curves(values, max_points=16, max_error=0.005)

    for name, points in zip(('Red', 'Green', 'Blue'), splines):
        points = np.rint(points * 255).astype(int)
        channels.append((f'ToneCurvePV2012{name}', points))

    for key, points in channels:
        seq = ET.SubElement(ET.SubElement(desc, f'{{{CRS}}}{key}'), f'{{{RDF}}}Seq')
        for x, y in points:
            ET.SubElement(seq, f'{{{RDF}}}li').text = f'{x}, {y}'

    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

    return path
