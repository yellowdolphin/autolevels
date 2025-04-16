#!/usr/bin/env python3
__version__ = '1.1.0'

from pathlib import Path
from argparse import ArgumentParser
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageFilter, ImageCms

import cv2
import piexif


KEEP_WHITE = False  # keep white instead of whitepoint if no whitepoint is specified
DEFAULT_QUALITY = 90
REPRODUCIBLE = {'blackpoint', 'whitepoint', 'blackclip', 'whiteclip', 'maxblack', 'minwhite', 'max_blackshift',
                'max_whiteshift', 'mode', 'gamma', 'model', 'saturation', 'saturation_first', 'saturation_before_gamma'}


def get_parser():
    """Return an argument parser for the CLI."""
    parser = ArgumentParser(description='Example: autolevels --blackpoint 10 --whitepoint 255 --gamma 1.2 input.jpg')

    points = parser.add_argument_group('Black and white point correction')
    points.add_argument(
        '--blackpoint', nargs='+', default=14, type=int, help=(
                                'Target black point, one L or three RGB values, range 0...255 (default: 14). '
                                'The image black point is measured per channel. If it is higher than the target '
                                'value, it will be lowered, otherwise kept unchanged.'))
    points.add_argument(
        '--whitepoint', nargs='+', default=None, type=int, help=(
                                'Target white point, one L or three RGB values, range 0...255 (default: keep). '
                                'The image white point is not changed by default, and will be increased to '
                                'this target value, if specified. '
                                'Note that hue and saturation of the highlights will be preserved (see MINWHITE). '
                                'Therefore, the target values may be lower for some channels.'))

    clipping = parser.add_argument_group('Clip shadows and highlights')
    # adobe auto color defaults are between 0.001 and 0.005
    clipping.add_argument(
        '--blackclip', nargs='+', default=0.002, type=float, help=(
                                'Percentage of pixels darker than black point (shadows clipped). '
                                'Due to noise and sharpening, the mathematical black point '
                                'can be lower than the perceived one. To mitigate this, the image black '
                                'point can be lowered until a certain percentage of the pixels is darker '
                                'than the set target value for the black point. The default of 0.002 '
                                'ignores the darkest 0.2 percent of the pixels when calculating the '
                                'current black point.'))
    clipping.add_argument(
        '--whiteclip', nargs='+', default=0.001, type=float, help=(
                                'Percentage of pixels brighter than white point (highlights clipped). '
                                'The default of 0.001 ingores the brightest 0.1 percent of the pixels when '
                                'calculating the current white point.'))

    limit = parser.add_argument_group('If the image is low in contrast, limit the correction of shadows and highlights')
    limit.add_argument(
        '--max-blackshift', nargs='+', default=30, type=int, help=(
                                'Upper limit for the black point shift (default: 30). '))
    limit.add_argument(
        '--max-whiteshift', nargs='+', default=30, type=int, help=(
                                'Upper limit for the white point shift (default: 30). Note that hue and '
                                'saturation of the highlights will be preserved (see MINWHITE). Therefore, '
                                'the shift can be lower for some channels.'))
    limit.add_argument(
        '--maxblack', nargs='+', default=None, type=int, help=(
                                'Extends the range where black points are fully corrected. If the current black '
                                'point is higher than MAXBLACK, the shift drops to MAX_BLACKSHIFT. By default, '
                                'the range is not extended and MAXBLACK = BLACKPOINT + MAX_BLACKSHIFT.'))
    limit.add_argument(
        '--minwhite', nargs='+', default=240, type=int, help=(
                                'Minimum white point (L or RGB values) that will be fully corrected to assume '
                                'WHITEPOINT. If the image white point is below MINWHITE, its hue and saturation '
                                'will be preserved, instead. Default: 240.'))

    bp_mode = parser.add_argument_group('Mode for determining the black and white points')
    bp_mode.add_argument(
        '--mode', default='hist', choices=['smooth', 'smoother', 'hist', 'perceptive'], help=(
                                'Black/white point sample mode: '
                                '"smooth" takes the pixel min/max values from a copy of the image smoothened with '
                                'a 3x3 envelope to compensate for noise/sharpen effects. '
                                '"smoother" does the same with a 5x5 envelope. '
                                '"hist" calculates the values at which a fraction of BLACKCLIP (WHITECLIP) '
                                'pixels is darker (brighter) than the black (white) point, respectively. '
                                '"perceptive" does the same with a weighted histogram, which is slower but can '
                                'improve the blackpoint of images with color cast. Use --simulate to check the '
                                'measured black and white points before and after processing.'))

    curve = parser.add_argument_group('Curve corrections')
    curve.add_argument(
        '--gamma', nargs='+', type=float, default=[1.0], help=(
                                'Gamma correction with inverse gamma (larger is brighter), one L or three RGB values '
                                '(default: 1.0).'))
    curve.add_argument(
        '--model', nargs='+', action='store', help='Model file(s) for free-curve correction')

    sat = parser.add_argument_group('Saturation')
    sat.add_argument(
        '--saturation', default=1, type=float, help=(
                                'A value of 0 produces a gray image, a value larger than 1 increases saturation '
                                '(default: 1.0).'))
    sat.add_argument(
        '--saturation-first', action='store_true', help='Adjust saturation before anything else')
    sat.add_argument(
        '--saturation-before-gamma', action='store_true', help='Adjust saturation before gamma (deprecated)')

    file_location = parser.add_argument_group('File locations')
    file_location.add_argument(
        '--folder', default='.', help='Path to input images')
    file_location.add_argument(
        '--prefix', default='', help='Common prefix of all input file names')
    file_location.add_argument(
        '--suffix', default='', help='Common suffix (including file extension) of all input file names')
    file_location.add_argument(
        '--fstring', default=None, help=(
                                'Expand input file names using a Python f-string. '
                                'Example: --fstring f"IMG_{x:04d}.jpg" -- 3 4 5  expands to  '
                                'IMG_0003.jpg IMG_0004.jpg IMG_0005.jpg'))
    file_location.add_argument(
        '--outdir', '--outfolder', default=None, help='Write output files here (default: current directory)')
    file_location.add_argument(
        '--outsuffix', default=None, type=str, help=(
                                'Suffix (including file extension) used in output file names. '
                                'Default: append "_al" to input file name (before file extension). '
                                'If both SUFFIX and OUTSUFFIX are specified, OUTSUFFIX replaces '
                                'SUFFIX in the output file name.'))
    file_location.add_argument(
        '--outprefix', default=None, type=str, help=(
                                'Prefix used in output file names. Default: none or same as input file name. '
                                'If both PREFIX and OUTPREFIX are specified, OUTPREFIX replaces SUFFIX in '
                                'the output file name.'))
    file_location.add_argument(
        '--outfstring', default=None, type=str, help=(
                                'If input file names are expanded using a Python f-string FSTRING, an '
                                'alternative f-string can be specified here for output files. Otherwise, '
                                'this option will be ignored.'))

    parser.add_argument(
        '--simulate', '--sandbox', action='store_true', help='Dry run: only read and process, skip file output')
    parser.add_argument(
        '--reproduce', default='', help=(
                                'Read CLI options from metadata of specified image REPRODUCE. '
                                'The latter must be the output of a compatible program version. '
                                'Example: autolevels --reproduce processed_image.jpg '
                                'other_images/*.jpg'))
    parser.add_argument(
        '--icc-profile', '--icc', default=None, type=str, help=(
                                'Specify ICC file for input image(s). If provided, color space will be '
                                'converted to sRGB after all corrections. This feature disables 48-bit output.'))
    parser.add_argument(
        '--reset-icc', action='store_true', help=(
                                'The input image is first converted from sRGB to the profile specified with the '
                                '--icc-profile option, then all corrections are applied, then the image is converted '
                                'back to sRGB before saving. This feature disables 48-bit processing.'))
    parser.add_argument(
        '--version', action='store_true', help='Print version information and exit')

    parser.add_argument(
        'files', nargs='*', action='store', help='Input files to process. Example: scans/IMG_*.jpg')

    return parser


def extract_arg(filename, parser):
    """Extract args from a previous autolevel output file"""

    old_namespace = parser.parse_args()

    filename = Path(filename)
    if not filename.exists():
        return f'Error: no file {filename}'

    img = Image.open(filename)

    cli_params = ''

    # parse JPEG comment
    if hasattr(img, 'info') and 'comment' in img.info:
        comment = img.info['comment'].decode()
        comment = comment.split('\n')[-1]  # multiline: read only last
        if 'autolevels ' in comment:
            version = comment.split('autolevels ')[1].split(',')[0]
            if version != __version__:
                print(f'WARNING: autolevels version changed: {version} -> {__version__}')
            cli_params += comment.split('params: ')[1]

    # parse CLI args
    new_namespace = parser.parse_args(cli_params.split())
    new_namespace.files = old_namespace.files
    new_namespace.cli_params = cli_params
    assert hasattr(new_namespace, 'cli_params'), 'WTF'
    return new_namespace


def merge_args(*, current_arg, extracted_arg):
    """Returns updated `current_arg` Namespace with reproducible parameters from `extracted_arg`."""
    for name in REPRODUCIBLE:
        if name not in current_arg:
            return f'WARNING: foreign parameter {name}, update reproducible_params!'
        setattr(current_arg, name, getattr(extracted_arg, name))
    current_arg.cli_params = extracted_arg.cli_params
    return current_arg


def evaluate_fstring(s: str, x):
    """
    Safely evaluates a string containing a Python f-string with a variable.

    Args:
        s (str): A string representing an f-string (e.g., "f'IMG_{fn:04d}.jpg'")
        x: The value to substitute (can be str or int)

    Returns:
        str: The evaluated string

    Raises:
        ValueError: If the input is not a valid f-string or no valid variable found
    """
    import re

    # Sanitize f-string for common argparse issues
    assert len(s) > 1, 'The f-string is improperly formatted or missing quotes'
    s = s[1:] if (s[0] == 'f') else s  # remove leading "f"
    s = '"' + s + '"' if (s[0] not in {'"', "'"}) else s  # add missing quotes
    s = 'f' + s if ('{' in s) else s  # add leading "f" if required

    # Checks on the f-string: starts with "f", quotes, len limits
    if not all([len(s) >= 5, len(s) < 1000, s.endswith(s[1]), s.count('{') <= 1, s.count('}') <= 1]):
        raise ValueError('The f-string is improperly formatted or missing quotes')

    # Find all variable patterns in the f-string, allowing whitespace around the variable name
    matches = re.findall(r'\{\s*(\w+)\s*(:[^}]*)?\}', s)
    if len(matches) == 0:
        raise ValueError('No valid variable symbol found in the f-string')
    elif len(matches) > 1:
        raise ValueError('The f-string contains more than one variable symbol')

    # Extract the single variable name and its specifier
    var_name, specifier = matches[0]
    var_name = var_name.strip()  # Remove any surrounding whitespace
    specifier = specifier if specifier else ''

    # Check if the variable name is a valid Python identifier
    if not var_name.isidentifier():
        raise ValueError(f'Invalid variable name "{var_name}" in the f-string')

    # Check specifier does not contain huge numbers
    specifier_numbers = re.findall(pattern=r'\d+', string=specifier)
    if any(int(n) > 500 for n in specifier_numbers):
        raise ValueError('The f-string is improperly formatted or missing quotes')

    # Check if the specifier ends with 'd' and convert x to int if so
    if specifier.endswith('d'):
        try:
            x = int(x)
        except ValueError:
            raise ValueError(f'Format is "d", but "{x}" is not a number')

    # Remove the f-string prefix and surrounding quotes, replace with format-compatible syntax
    formatted_str = re.sub(r'\{\s*(\w+)\s*(:[^}]*)?\}', r'{' + specifier + '}', s[2:-1])

    # Safely format the string using str.format
    result = formatted_str.format(x)
    return result


def get_channel_cutoff(hist, thresh, upper=False, norm=None):
    """Return `hist` bin where accumulated count exceeds fraction of `thresh`.

    Args:
        hist (list or array-like): Histogram data.
        pixel_thresh (float): Fraction of the total count to reach.
        upper (bool, optional): If True, start accumulating from the last bin (descending order). Default: False.
        norm (int, optional): Normalize histogram with `norm` rather than sum(hist).

    Returns:
        int: The index of the bin where the accumulated count first exceeds `pixel_thresh`.
    """
    n_bins = len(hist)
    n_total = norm or sum(hist)
    limit = n_total * thresh
    accsum = 0
    _range = range(n_bins - 1, -1, -1) if upper else range(n_bins)
    for bin in _range:
        accsum += hist[bin]
        if accsum > limit:
            return bin


def get_blackpoint_whitepoint(array, maxvalue, mode, pixel_black, pixel_white):
    """Returns black point and white point

    uint16 images are converted to uint8 for fast histogram evaluation.
    """
    # 3x3 or 5x5 envelope
    SMOOTH = ImageFilter.SMOOTH_MORE if mode == 'smoother' else ImageFilter.SMOOTH

    # convert to PIL.Image
    if array.dtype == np.dtype('uint16'):
        img = Image.fromarray((array.astype('float32') * (255 / 65535)).clip(0, 255).astype('uint8'))
    elif array.dtype in {np.dtype('float32'), np.dtype('float64')}:
        img = Image.fromarray((array * (255 / maxvalue)).clip(0, 255).astype('uint8'))
    elif array.dtype == np.dtype('uint8'):
        img = Image.fromarray(array)

    if (mode == 'perceptive') and (img.mode == 'L'):
        mode = 'hist'  # equivalent for gray scale images

    if mode.startswith('smooth'):
        img = img.filter(SMOOTH)
        array = np.array(img)  # HWC
        return array.min(axis=(0, 1)), array.max(axis=(0, 1))

    elif mode == 'perceptive_serial':
        if img.mode != 'RGB':
            return f'Error: image mode "{img.mode}" not supported by perceptive sampling mode'
        R, G, B = array.transpose(2, 0, 1)
        L = np.array(img.convert(mode='L'), dtype=np.float32)  # faster than np.mean or python
        L_bp = L.min()
        L = (L - L_bp) * (255 / (255 - L_bp)) + 0.5
        pixel_black = pixel_black if pixel_black.shape == (3,) else pixel_black.repeat(3)
        pixel_white = pixel_white if pixel_white.shape == (3,) else pixel_white.repeat(3)
        n_pixel = img.height * img.width
        blackpoint, whitepoint = [], []

        for pix_black, pix_white, channel in zip(pixel_black, pixel_white, (R, G, B)):
            weight = np.where(channel >= L, 1, channel / L)

            # this is the bottleneck (uint8: even slower)
            hist, _ = np.histogram(channel, bins=256, range=(0, 256), weights=weight)

            # the rest takes no time
            blackpoint.append(get_channel_cutoff(hist, thresh=pix_black, upper=False, norm=n_pixel))
            whitepoint.append(get_channel_cutoff(hist, thresh=pix_white, upper=True, norm=n_pixel))

        return np.array(blackpoint), np.array(whitepoint)

    elif mode == 'perceptive':
        if img.mode != 'RGB':
            return f'Error: image mode "{img.mode}" not supported by perceptive sampling mode'
        L = np.array(img.convert(mode='L'), dtype=np.float32)  # faster than np.mean or python
        L_bp = L.min()
        L = (L - L_bp) * (255 / (255 - L_bp)) + 0.5
        pixel_black = pixel_black if pixel_black.shape == (3,) else pixel_black.repeat(3)
        pixel_white = pixel_white if pixel_white.shape == (3,) else pixel_white.repeat(3)
        n_pixel = img.height * img.width
        blackpoint, whitepoint = [], []

        # Process RGB channels in parallel because numpy weighted histogram is slow
        R, G, B = img.split()[:3]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_channel, pix_black, pix_white, channel, L, n_pixel)
                for pix_black, pix_white, channel in zip(pixel_black, pixel_white, (R, G, B))]

            # Gather results in the order of submission
            for future in futures:
                black, white = future.result()
                blackpoint.append(black)
                whitepoint.append(white)

        return np.array(blackpoint), np.array(whitepoint)

    elif mode in {'hist', 'histogram'}:
        channels = img.split()
        pixel_black = pixel_black if pixel_black.shape == (3,) else pixel_black.repeat(len(channels))
        pixel_white = pixel_white if pixel_white.shape == (3,) else pixel_white.repeat(len(channels))

        n_pixel = img.height * img.width
        blackpoint, whitepoint = [], []

        for pix_black, pix_white, channel in zip(pixel_black, pixel_white, channels):
            hist = channel.histogram()

            blackpoint.append(get_channel_cutoff(hist, thresh=pix_black, upper=False))
            whitepoint.append(get_channel_cutoff(hist, thresh=pix_white, upper=True))

        return np.array(blackpoint), np.array(whitepoint)

    elif mode == 'hist2':
        # Deprecated.
        # More concise but not faster than hist (btw, np.percentile is 5 x times slower).
        channels = img.split()
        pixel_black = pixel_black if pixel_black.shape == (3,) else pixel_black.repeat(len(channels))
        pixel_white = pixel_white if pixel_white.shape == (3,) else pixel_white.repeat(len(channels))

        n_pixel = img.height * img.width
        blackpoint, whitepoint = [], []

        for pix_black, pix_white, channel in zip(pixel_black, pixel_white, channels):
            hist = np.array(channel.histogram()) / n_pixel

            cumsum = np.cumsum(hist)
            blackpoint.append(np.argmax(cumsum > pix_black))
            whitepoint.append(np.argmax(cumsum > (1 - pix_white)))

        return np.array(blackpoint), np.array(whitepoint)


def process_channel(pix_black, pix_white, channel, L, norm=None):
    weight = np.where(channel >= L, 1, channel / L)
    hist, _ = np.histogram(channel, bins=256, range=(0, 256), weights=weight)
    norm = norm or channel.shape[0] * channel.shape[1]

    # Calculate blackpoint and whitepoint for this channel
    blackpoint = get_channel_cutoff(hist, thresh=pix_black, upper=False, norm=norm)
    whitepoint = get_channel_cutoff(hist, thresh=pix_white, upper=True, norm=norm)

    return blackpoint, whitepoint


def blend(a, b, alpha=1.0):
    """Interpolate between arrays `a`and `b`"""
    return a if (alpha == 1) else alpha * a + (1.0 - alpha) * b


def grayscale(rgb, mode='itu', keep_channels=False):
    """Convert RGB image (float array) to L"""

    if mode == 'itu':
        # Rec. ITU-R BT.601-7 definition of luminance
        R, G, B = (rgb[:, :, c] for c in range(3))
        L = R * 0.299 + G * 0.587 + B * 0.114
    elif mode == 'mean':
        L = rgb.mean(axis=2)
    else:
        raise ValueError(f'mode {mode} not supported')

    return np.stack([L, L, L]) if keep_channels else L[:, :, None]


def get_out_format(filename, pil_img):
    """Infer format from filename extension or use input format"""
    ext = Path(filename).suffix.lower()
    pil_extensions = Image.registered_extensions()
    return pil_extensions.get(ext, pil_img.format)


def estimate_jpeg_quality(pil_img):
    """Infer quality from qantization table if found, else return default quality."""
    if not hasattr(pil_img, 'quantization') or pil_img.quantization is None:
        return DEFAULT_QUALITY
    qtable = pil_img.quantization[0]
    max_q = 100
    m = 1.15
    return round(max_q - np.mean(qtable) / m)


def purge_cli_params(args, fn):
    """Return a str of all args required for the --reproduce feature"""
    impossible_args = {'--simulate', '--sandbox', '--reproduce'}  # if present, purge shouldn't be called
    iterator = iter(args)
    cli_params = []
    while True:
        try:
            param = next(iterator)
            if param == '--':
                cli_params.append(param)
                cli_params.append(fn.name)  # save original file name but not its path
                break
            if param.startswith('--'):
                assert param not in impossible_args, f'{param} outside expected scope - this is a bug!'
                var_name = param[2:].replace('-', '_')
                if var_name not in REPRODUCIBLE:
                    _ = next(iterator)  # all non-reproducible params are key-value pairs
                    continue
            cli_params.append(param)
        except StopIteration:
            break

    return ' '.join(cli_params)


def make_comment(img, version, cli_params):
    """Save program version and CLI parameters in JPEG comment or EXIF"""

    comments = []

    # Keep existing comments
    if hasattr(img, 'info') and 'comment' in img.info:
        try:
            comments.append(img.info['comment'].decode())
        except UnicodeDecodeError:
            pass  # drop non-text comments

    comments.append(f'autolevels {version}, params: {cli_params}')

    return '\n'.join(comments)


def main(callback=None):
    """Pass callback when processing multiple files with a curve model.

    callback (callable): call when finishing a file, pass input_path (str), True, info_str
    If error occurs: pass input_path (str), False, error message (str) to proceed or
    return an error message to abort.
    """
    parser = get_parser()
    arg = parser.parse_args()

    if arg.version:
        print(f'AutoLevels version {__version__}')
        return

    if not arg.files:
        parser.print_usage()
        return 'No files specified'

    # Post-process arg
    if arg.reproduce:
        extracted_arg = extract_arg(arg.reproduce, parser)
        assert hasattr(extracted_arg, 'cli_params'), 'called extract_arg but arg has no cli_params'
        arg = merge_args(current_arg=arg, extracted_arg=extracted_arg)
        assert hasattr(arg, 'cli_params'), 'merge_args deleted cli_params'
        print(f'Reproducing {arg.reproduce} processing: {arg.cli_params}')
    sample_mode = arg.mode
    blackclip = np.array(arg.blackclip, dtype=float)
    max_blackshift = np.array(arg.max_blackshift, dtype=int)
    whiteclip = np.array(arg.whiteclip, dtype=float)
    min_white = np.array(arg.minwhite, dtype=int)
    max_whiteshift = np.array(arg.max_whiteshift, dtype=int)
    if not all(g > 0 for g in arg.gamma):
        return f'Error: invalid gamma {arg.gamma}, must be positive'
    gamma = 1 / np.array(arg.gamma, dtype=float)
    if arg.model:
        for fn in arg.model:
            if not Path(fn).exists():
                return f'Error: Specified model file could not be found: {fn}'
    if arg.icc_profile:
        icc_file = Path(arg.icc_profile)
        if not icc_file.exists():
            return f'Error: file not found: {icc_file}'
        icc_profile = ImageCms.getOpenProfile(str(icc_file))
        sRGB_profile = ImageCms.createProfile('sRGB')

    # Input file names
    path = Path(arg.folder)
    if not path.exists():
        return f'Error: folder "{path}" does not exist.'
    pre = arg.prefix
    if pre.startswith(('.', '/')):
        return f'Error: unsecure prefix "{pre}", use --folder to specify the path'
    suf = arg.suffix

    if arg.fstring:
        fns = [path / evaluate_fstring(arg.fstring, x) for x in arg.files]
        # Check input files exist (fail early)
        for fn in fns:
            assert fn.exists(), f'File not found: {fn}'
    else:
        # Use prefix, suffix, and shell/glob expansion
        fns = []
        for x in arg.files:
            if x in {'.', '..', '/'}:
                print(f'Skipping "{x}"')
                continue
            try:
                parent, stem, ext = Path(x).parent, Path(x).stem, Path(x).suffix
                name = f'{pre}{stem}{suf}{ext}'
                glob_pattern = name
                matches = sorted((path / parent).glob(glob_pattern))
                if len(matches) > 0:
                    fns.extend(matches)
                else:
                    fns.append(path / parent / name)
            except Exception as e:
                print(e)
                return f'No matching files found for {x}'

    if not fns:
        return f'No matching files found in "{path}"'

    # Output file options
    outdir = Path(arg.outdir) if arg.outdir else Path('.')
    if outdir and not arg.simulate:
        outdir.mkdir(exist_ok=True)

    # TODO: implement batchwise inference on fns
    if arg.model:
        # Free-curve correction from predicted curve
        from .inference import get_model, get_ensemble, free_curve_map_image

        if len(arg.model) == 1:
            model = get_model(arg.model[0])
        else:
            model = get_ensemble(arg.model)

    # Process input files
    for i, fn in enumerate(fns):
        # Skip non-existing
        if not fn.exists():
            print(f"Error: {fn} not found - skipping")
            if callback is not None:
                callback(str(fn), False, f'Error: {fn} not found - skipping')
            continue

        # Decide output file name
        if arg.outfstring:
            out_fn = outdir / evaluate_fstring(arg.outfstring, arg.files[i])
        else:
            stem, ext = fn.stem, fn.suffix
            if arg.outprefix and arg.prefix:
                stem = stem.replace(arg.prefix, arg.outprefix, 1)
            if arg.outsuffix and arg.suffix:
                # Find index where suffix starts so we can replace it
                start = len(arg.outprefix or '')
                start = fn.name.rfind(arg.suffix, start)
                stem = stem[:start]  # strip arg.suffix
                suf = arg.outsuffix
            else:
                suf = arg.outsuffix or f'_al{ext}'
            out_fn = (outdir or fn.parent) / f'{stem}{suf}'
        # TODO: check out_fn exists, add option -f to overwrite

        # Open image if possible
        try:
            pil_img = Image.open(fn)
        except Exception as e:
            print(f'Error: skipping {fn}, {e}')
            if callback is not None:
                callback(str(fn), False, f'Error: broken or unsupported image format (skipping) {fn}')
            continue

        if arg.icc_profile and arg.reset_icc:
            # Convert from sRGB to ICC profile
            try:
                array = np.array(ImageCms.profileToProfile(pil_img, sRGB_profile, icc_profile))
            except ImageCms.PyCMSError as e:
                print(e, "ICC probably has no B2A")
                return "This ICC profile cannot be used for reverse transformation."
        else:
            array = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        maxvalue = 65535 if array.dtype == np.dtype('uint16') else 255

        # Check conditions for 48-bit output
        out_48bit = all((array.dtype == np.dtype('uint16'),
                         out_fn.suffix.lower()[1:4] in {'png', 'tif'},
                         arg.icc_profile is None))

        # Handle image modes
        img_alpha = None
        if pil_img.mode == 'RGBA':
            img_alpha = pil_img.getchannel('A')
            transparency = np.array(img_alpha).min() < 255
            if transparency:
                if maxvalue > 255:
                    print("Warning: this is an RGBA image with transparency. "
                          "Flatening to canvas is necessary for any corrections, "
                          "depth will be lowered to 8-bit. Assuming white canvas.")
                else:
                    print("Warning: this is an RGBA image with transparency, assuming white canvas.")
                r, g, b, img_alpha = pil_img.split()
                canvas = Image.new('RGB', pil_img.size, (255, 255, 255))
                canvas.paste(pil_img, mask=img_alpha)
                array = np.array(canvas)
                out_48bit = False
                del canvas, r, g, b
            else:
                # discard empty alpha channel
                img_alpha = None
        if (pil_img.mode == 'L') and (arg.saturation != 1):
            print(f'Warning: "{fn}" is gray scale image, ignoring saturation options.')
            saturation = 1
        else:
            saturation = arg.saturation

        # Adjust saturation before anything else
        if (saturation != 1) and arg.saturation_first:
            L = grayscale(array)
            array = blend(array, L, saturation)

        if arg.model:
            # Simulate: just test inference on first image
            if arg.simulate and fn != fns[0]:
                print(f'{fn} -> {out_fn}')
                continue

            resized = cv2.resize(array, (384, 384)[::-1])  # uint16 or uint8
            free_curve = model(resized)

            array = free_curve_map_image(array, free_curve)  # float32, range (0, 1)

            if arg.simulate:
                print(f'{fn} -> {out_fn}')
                continue

        else:
            blackpoint, whitepoint = get_blackpoint_whitepoint(array, maxvalue, sample_mode, blackclip, whiteclip)

            # Set targets, limit shifts in black/white point for low-contrast images
            target_black = np.array(arg.blackpoint, dtype=int)
            target_white = np.array(arg.whitepoint, dtype=int) if arg.whitepoint else None
            max_black = target_black + max_blackshift if arg.maxblack is None else np.array(arg.maxblack, dtype=int)
            if (blackpoint > max_black).any():
                target_black = np.maximum(target_black, blackpoint - max_blackshift)
            if (whitepoint < min_white).any() and arg.whitepoint:
                if np.var(max_whiteshift) == 0:
                    # avoid clipping to preserve hue + saturation of white point
                    max_whiteshift = np.minimum(max_whiteshift, (target_white - whitepoint).min())
                shift = max_whiteshift * whitepoint / (255 - max_whiteshift)  # stay below max_whiteshift
                target_white = np.minimum(target_white, whitepoint + shift)
            elif arg.whitepoint:
                target_white = np.minimum(target_white, whitepoint + max_whiteshift)  # stay below max_whiteshift in any case

            # Set black point to min(target_black, blackpoint).
            target_black = np.minimum(target_black, blackpoint)

            # Set white point to max(target_white, whitepoint) or preserve it.
            if KEEP_WHITE and (target_white is None):
                whitepoint = np.array([255, 255, 255])
            target_white = whitepoint if target_white is None else np.maximum(target_white, whitepoint)

            # Simulate: just print black and white points
            if arg.simulate:
                print(f'{fn} -> {out_fn} (black point: {blackpoint} -> {target_black.round().astype("int")}, '
                      f'white point: {whitepoint} -> {target_white.round().astype("int")})')
                continue

            # Make target black/white points gamma-agnostic
            black = 255 * np.power(target_black / 255, 1 / gamma)
            white = 255 * np.power(target_white / 255, 1 / gamma)

            shift = (blackpoint - black) * white / (white - black)
            stretch_factor = white / (whitepoint - shift)

            array = array.astype(np.float32)
            array = (array - shift * (maxvalue / 255)) * stretch_factor
            if (shift < 0).any():
                # small gamma results in a low black point => upper limit for target_black!
                channels = [name for name, s in zip('RGB', shift) if s < 0]
                print(f'{fn} WARNING: lower black point or increase gamma for channel(s)', *channels)

            array /= maxvalue  # range (0, 1)

        # Adjust saturation before gamma (deprecated)
        if (saturation != 1 and arg.saturation_before_gamma and not arg.saturation_first):
            L = grayscale(array)
            array = blend(array, L, saturation)

        # Gamma correction
        if (gamma != 1).any():
            array = array.clip(0, None)
            array = np.power(array, gamma)

        # Adjust saturation
        if (saturation != 1 and not (arg.saturation_before_gamma or arg.saturation_first)):
            L = grayscale(array)
            array = blend(array, L, saturation)

        if out_48bit:
            array = (array * 65535).round().clip(0, 65535).astype('uint16')
            cv2.imwrite(out_fn, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        else:
            array = (array * 255).round().clip(0, 255).astype('uint8')

            img = Image.fromarray(array)

            # Convert color space from ICC profile to sRGB
            if arg.icc_profile:
                img = ImageCms.profileToProfile(img, icc_profile, sRGB_profile)

            # Merge with alpha (RGBA images only)
            if img_alpha is not None:
                img = Image.merge('RGBA', [*img.split(), img_alpha])

            # Add attributes required to preserve JPEG quality
            for attr in 'format layer layers quantization'.split():
                value = getattr(pil_img, attr, None)
                if value is not None:
                    setattr(img, attr, value)

            # Update info attribute (PIL.Image always has one) and transfer to new image
            pil_img.info.update(img.info)
            setattr(img, 'info', pil_img.info)

            # Configure save options
            kwargs = {}
            out_format = get_out_format(out_fn, img)
            if out_format in {'JPEG'}:
                kwargs['quality'] = 'keep' if hasattr(img, 'quantization') else DEFAULT_QUALITY
            elif (img.format == 'JPEG') and out_format in {'TIFF'}:
                # Try to preserve input image JPEG quality empirically (TIFF files are larger)
                kwargs['compression'] = 'jpeg'
                jpeg_quality = estimate_jpeg_quality(img)
                kwargs['quality'] = max(24, 24 + round((jpeg_quality - 44) * (100 - 24) / (100 - 44)))
            elif out_format == 'TIFF':
                # Keep input image compression if available
                kwargs['compression'] = img.info.get('compression', 'raw')
            if 'icc_profile' in img.info and out_format in {'JPEG'}:
                kwargs['icc_profile'] = img.info.get('icc_profile')
            if 'dpi' in img.info and out_format in {'JPEG', 'TIFF', 'PNG'}:
                kwargs['dpi'] = tuple(round(x) for x in img.info['dpi'])

            # Make reproducible, leave CLI args in JPEG comment
            if getattr(arg, 'cli_params', None):
                cli_params = arg.cli_params
            else:
                cli_params = purge_cli_params(sys.argv[1:], fn)
            comment = make_comment(pil_img, __version__, cli_params)

            try:
                # Let PIL derive file format from extension
                img.save(out_fn, comment=comment, optimize=True, **kwargs)
            except ValueError as e:
                # If that fails, save in original format
                print(f"{e}, saving in {img.format}.")
                img.save(out_fn, format=img.format, comment=comment, optimize=True, **kwargs)

            # Neither PIL nor piexif correctly decode the (proprietary) MakerNotes IFD.
            # Hence, this is the only way to fully preserve the entire EXIF:
            if 'exif' in pil_img.info:
                piexif.transplant(str(fn), str(out_fn))

        # Logging
        infos = [f'{fn} -> {out_fn}']
        if not arg.model and (blackpoint != target_black).any():
            high = 'high ' if (blackpoint > max_black).any() else ''
            infos.append(f'{high}black point: {blackpoint} -> {target_black.round().astype("int")}')
        if not arg.model and (whitepoint != target_white).any():
            low = 'low ' if (whitepoint < min_white).any() else ''
            infos.append(f'{low}white point: {whitepoint} -> {target_white.round().astype("int")}')
        print(', '.join(infos))

        # Callback
        if callback is not None:
            callback(str(fn), True, infos)


if __name__ == '__main__':
    main()
