"""Photoshop ACV version 4: composite identity plus separate R/G/B curves."""
from pathlib import Path
import struct

import numpy as np

from .common import prepare_curves
from .darktable import fit_rgb_curves


def export_curves(path, curves):
    """Write big-endian ACV data, at most 19 points/curve, output before input.

    Coordinates use the ACV 0..255 scale even for higher-bit-depth images.
    Point reduction, quantization and Photoshop interpolation are approximate.
    """
    values = prepare_curves(curves)
    channels = [np.array([[0, 0], [255, 255]])]
    splines = fit_rgb_curves(values, max_points=19, max_error=0.005)
    channels.extend(np.rint(points * 255).astype(int) for points in splines)
    data = bytearray(struct.pack('>HH', 4, len(channels)))
    for points in channels:
        data.extend(struct.pack('>H', len(points)))
        for x, y in points:
            data.extend(struct.pack('>HH', int(y), int(x)))
    Path(path).write_bytes(data)
    return Path(path)
