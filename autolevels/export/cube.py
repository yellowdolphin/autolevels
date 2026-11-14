"""Iridas 1D CUBE LUTs retaining all 256 RGB samples."""
from pathlib import Path

from .common import prepare_curves


def export_curves(path, curves):
    """Write a normalized-domain 1D LUT, one R/G/B triplet per input level.

    Use linear interpolation in the consumer to match AutoLevels. The file
    contains no ICC profile or color-space conversion and no 3D LUT table.
    """
    values = prepare_curves(curves)
    lines = ['TITLE "AutoLevels RGB curves"', 'LUT_1D_SIZE 256',
             'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0']
    lines.extend(' '.join(f'{v:.10f}' for v in row) for row in values.T)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return Path(path)
