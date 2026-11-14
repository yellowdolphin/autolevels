"""Shared validation and dispatch for standalone curve exports."""
from importlib import import_module
from pathlib import Path

import numpy as np


# Standalone exporters: CLI argument == module name; value is the filename suffix.
# darktable uses its separate XMP-history export path in autolevels.py.
EXPORT_FORMATS = {
    'lightroom': '.lightroom.xmp',
    'rawtherapee': '.pp3',
    'acv': '.acv',
    'cube': '.cube',
}


def prepare_curves(curves):
    """Copy channel-major samples (3, 256) or (768,), with singleton wrappers.

    Reject malformed/nonfinite data and clamp outputs to the image's [0, 1]
    range. No profile conversion is performed by an exporter.
    """
    values = np.asarray(curves, dtype=np.float64)
    if values.squeeze().shape not in ((3, 256), (768,)):
        raise ValueError(f'Expected RGB curves with 256 samples per channel, got {values.shape}')
    if not np.isfinite(values).all():
        raise ValueError('Curve samples must be finite')
    return values.reshape(3, 256).clip(0, 1)


def export_curves(format_name, output_image, curves, *, grayscale=False, simulate=False):
    """Write a companion to the output image, returning its path.

    Appending the suffix preserves the image extension, avoiding collisions
    between e.g. scan.jpg and scan.tif. Adobe presets never use the ordinary
    image.xmp sidecar name. Existing companions are replaced, like image output.
    """
    output_image = str(output_image).removesuffix(".lightroom.xmp")
    path = Path(output_image).with_suffix(EXPORT_FORMATS[format_name])
    if not simulate:
        values = prepare_curves(curves)
        if grayscale:
            # Match free_curve_map_image's averaged correction for gray input.
            values = np.repeat(values.mean(axis=0, keepdims=True), 3, axis=0)
        exporter = import_module(f'.{format_name}', __package__)
        exporter.export_curves(path, values)
    print(f'Export {format_name}: {path}')
    return path
