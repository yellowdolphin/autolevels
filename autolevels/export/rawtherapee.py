"""RawTherapee partial processing profiles RGB Curves export."""
import os
from pathlib import Path

from .common import prepare_curves
from .darktable import fit_rgb_curves

PP3_VERSION = 329
MAX_FIT_ERROR = 0.005
CHANNEL_KEYS = ('rCurve', 'gCurve', 'bCurve')


def export_curves(path, curves):
    """Write fitted Hermite nodes as Flexible/CatmullRom curves (type 4).

    Use the existing darktable fitter: up to 20 points/channel and a 0.005
    normalized fitting-error target. RawTherapee reconstructs the spline from
    the nodes; PP3 does not store the fitter's tangents. Load in Preserve
    mode. Version 329 is the schema introducing the RGB Curves Enabled flag;
    only these long-established fields are used, not newer processing tools.
    """
    path = Path(path)
    splines = fit_rgb_curves(prepare_curves(curves), max_error=MAX_FIT_ERROR)

    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        lines = []

    _ensure_min_version(lines, PP3_VERSION)
    _replace_section(lines, 'RGB Curves', _rgb_curves_section(splines))

    _write_atomic(path, '\n'.join(lines) + '\n')
    return path


def section_bounds(lines, name):
    """Return the start and end line indices of a PP3 section."""
    header = f'[{name}]'
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == header),
        None,
    )
    if start is None:
        return None

    end = next(
        (i for i in range(start + 1, len(lines)) if _is_header(lines[i])),
        len(lines),
    )
    return start, end


def _is_header(line):
    line = line.strip()
    return line.startswith('[') and line.endswith(']')


def _curve_line(key, nodes):
    coords = ''.join(f'{v:.10f};' for v in nodes.flat)
    return f'{key}=4;{coords}'


def _rgb_curves_section(splines):
    curve_lines = [
        _curve_line(key, nodes)
        for key, nodes in zip(CHANNEL_KEYS, splines, strict=True)
    ]
    return ['[RGB Curves]', 'Enabled=true', 'LumaMode=false', *curve_lines, '']


def _ensure_min_version(lines, minimum):
    """Make sure [Version] exists and its Version key is >= `minimum`."""
    bounds = section_bounds(lines, 'Version')
    if bounds is None:
        lines[:0] = ['[Version]', f'Version={minimum}', '']
        return

    start, end = bounds
    for i in range(start + 1, end):
        key, _, value = lines[i].partition('=')
        if key.strip() == 'Version':
            try:
                current = int(value)
            except ValueError:
                current = 0
            lines[i] = f'Version={max(minimum, current)}'
            return

    # Section exists without a Version key: add it after the last content
    # line so the blank separator before the next section is preserved.
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, f'Version={minimum}')


def _replace_section(lines, name, section):
    """Replace section `name` in place, or append it if it doesn't exist."""
    bounds = section_bounds(lines, name)
    if bounds is not None:
        start, end = bounds
        lines[start:end] = section
        return

    if lines and lines[-1].strip():
        lines.append('')
    lines.extend(section)


def _write_atomic(path, text):
    """Write via temp file + rename so a failure can't truncate the PP3."""
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
