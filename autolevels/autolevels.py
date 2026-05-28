#!/usr/bin/env python3
__version__ = '1.4.0'

from pathlib import Path
from argparse import ArgumentParser
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

import cv2
from exiftool import ExifToolHelper
from exiftool.exceptions import ExifToolExecuteError

from autolevels.icc.icc import (get_icc_profile, convert_to_srgb, convert_curve, get_srgb_profile,
                                profile_to_profile, infer_gamma, convert_curve_gamma, convert_curve_profile)


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
        '--exiftool', default=None, type=str, help=(
                                'Path to exiftool executable. If not provided, will try to find it in PATH.'))
    parser.add_argument(
        '--skip-metadata', action='store_true', help=(
                                'Do not transfer any metadata from input to output image'))
    parser.add_argument(
        '--input-icc-profile', default=None, type=str, help=(
                                'Specify ICC profile file for input image(s). If not provided, '
                                'the embedded profile (if present) or sRGB will be used.'))
    parser.add_argument(
        '--output-icc-profile', default=None, type=str, help=(
                                'Specify ICC profile file for output image(s). If not provided, '
                                'the embedded profile (if present) or sRGB will be used.'))
    parser.add_argument(
        '--rendering-intent', default='perceptual', type=str, help=(
                                'Specify rendering intent for color conversion. '
                                'Options: perceptual, relative_colorimetric, saturation, absolute_colorimetric. '
                                'Default: perceptual.'))
    parser.add_argument(
        '--model-space', default='sRGB', type=str, help=(
                                'Color adaptation for model input. '
                                'Options: sRGB (default), TRC, gamma, none (stay in input space).'))
    parser.add_argument(
        '--lut-interpolation', default='linear', type=str, help=(
                                'LUT Interpolation method. Options: linear (default), tetrahedral (slower).'))

    parser.add_argument(
        '--export', nargs='+', action='store', default=None, help=(
            'Export curves to supported programs. '
            '"darktable": Append rgb curve to history of an existing darktable XMP file. You can skip image output '
            'by specifying an OUTSUFFIX or OUTFSTRING ending with ".xmp".'))

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

    with Image.open(filename) as img:
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


def imread_unicode(fn, flags=cv2.COLOR_BGR2RGB, src_bytes=None):
    """Unicode-safe cv2.imread replacement.

    Args:
        fn (str | Path): File name or Path
        flags (int, optional): OpenCV flags for image decoding. Default: cv2.COLOR_BGR2RGB
        src_bytes (bytes, optional): Bytes object with pixel data. Default: None

    Returns:
        np.ndarray: Decoded image array or str: Error message

    Handles all known errors, supports 16-bit images.
    """
    # The bytes object for decoding is always uint8, regardless of pixel depth
    if src_bytes is None:
        try:
            src_bytes = Path(fn).read_bytes()
        except Exception as e:
            return f"Could not read {fn}: {e}"
    array = np.frombuffer(src_bytes, np.uint8)

    if len(array) == 0:
        return f"{fn} is an empty file, no image data found"

    # Decode image and convert to RGB
    array = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

    if array is None:
        return f"cv2 could not decode {fn}"
    try:
        array = cv2.cvtColor(array, flags)
    except ValueError as e:
        return f"cv2 could not convert {fn} to RGB: {e}"
    return array


def imwrite_unicode(path, array, default_ext='.jpeg', params=None):
    """
    Unicode-safe replacement for cv2.imwrite().

    Args:
        path (str | Path): File name or Path
        array (np.ndarray): Image array
        default_ext (str, optional): File extension if path has no suffix. Default: '.jpeg'
        params (list, optional): OpenCV parameters for image encoding. Default: None

    Returns:
        bool: True if successful, False otherwise

    Works for any filename (UTF-8), and supports 8-bit and 16-bit images.
    """
    ext = Path(path).suffix or default_ext

    # Encode via OpenCV → returns a uint8 buffer containing PNG/JPEG/TIFF/etc. file bytes
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(ext, array, params or [])
    if not success:
        print(f"Could not encode {path}")
        return False

    # Write encoded bytes using Python’s Unicode-aware file I/O
    try:
        Path(path).write_bytes(encoded.tobytes())
    except Exception as e:
        print(f"Could not write {path}: {e}")
        return False
    return True


def transfer_metadata(fn, out_fn, out_format, kwargs, exiftool_path,
                      input_icc_profile, output_icc_profile):
    """
    Transfer metadata from input file to output file using exiftool.

    Args:
        fn (str | Path): Input file path
        out_fn (Path): Output file path
        out_format (str): Output file format
        pil_img (PIL.Image): PIL image object
        exiftool_path (str | Path): Path to exiftool executable
        input_icc_profile (dict): source ICC profile
        output_icc_profile (dict): target ICC profile

    Returns:
        None
    """
    if not exiftool_path or not Path(exiftool_path).exists():
        print(f"exiftool not found, metadata is not preserved in {out_fn}.")
        return

    # Embed ICC profile when known
    if output_icc_profile and output_icc_profile['path'] is not None:
        icc_path = output_icc_profile['path']
        if output_icc_profile['path'].suffix.lower() in {'icc'}:
            icc_arg = f'-icc_profile<="{icc_path}"'  # from ICC file
        elif icc_path == fn:
            icc_arg = '-icc_profile<icc_profile'  # embedded, from input image
        else:
            icc_arg = f'-tagsfromfile "{icc_path}" -icc_profile<icc_profile'  # embedded, from a third image
    elif (output_icc_profile and input_icc_profile
          and output_icc_profile['description'] == input_icc_profile['description']):
        # copy ICC profile from input image
        icc_arg = '-icc_profile<icc_profile'
    elif input_icc_profile:
        print(f"Warning: no source file for sRGB v2 ICC profile (input image: {input_icc_profile['description']})")
        icc_arg = ''  # should only happen if user requests built-in "sRGB" (v2) output profile
    else:
        icc_arg = ''  # don't embed sRGB v2 by default when input image has no ICC profile, neither

    if out_format == 'TIFF' and kwargs.get('compression', '') == 'jpeg':
        # Injecting metadata to TIFF with JPEG compression is unsafe
        from autolevels.tiff_processor import exiftool_safe_transfer
        result = exiftool_safe_transfer(fn, out_fn, icc_arg, exiftool_path)
        if result is False:
            print(f"exiftool could not transfer all metadata to {out_fn}.")
        return

    exiftool_args = ['-overwrite_original', '-tagsfromfile', f'{fn}', '-all:all', icc_arg, f'{out_fn}']
    with ExifToolHelper(executable=exiftool_path) as et:
        try:
            et.execute(*[a.encode() for a in exiftool_args])

        except ExifToolExecuteError as e:
            print(f"exiftool error: {e}")
            print(f"    {e.stdout}")
            print(f"    {e.stderr}")

        except Exception as e:
            print(f"exiftool error: {e}")


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
    else:
        return 0 if upper else n_bins - 1


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
    """Calculate blackpoint, whitepoint for a single `channel`

    Args:
        pix_black (float): clipping threshold for black
        pix_white (float): clipping threshold for white
        channel (PIL.Image): color channel to process
        L (numpy.ndarray): Luminance
        norm (int, optional): Normalize histogram with `norm` rather than sum(hist).
    """
    weight = np.where(channel >= L, 1, channel / (L + 0.1))
    # hist, _ = np.histogram(channel, bins=256, range=(0, 256), weights=weight)
    # channel, L, weight have same shape, but on github actions (python 3.14), this raises
    # ValueError: operands could not be broadcast together with shapes (256,) (257,) (256,)
    # This workaround is 3 x faster:
    hist = np.bincount(np.asarray(channel).ravel(), weights=weight.ravel(), minlength=256)
    norm = norm or channel.size[0] * channel.size[1]

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
    """Infer quality from qantization table if found, else return default quality.

    pil_img: PIL Image or qtables (dict)
    """
    if isinstance(pil_img, dict):
        qtable = pil_img[0]
    elif hasattr(pil_img, 'quantization') and pil_img.quantization is not None:
        qtable = pil_img.quantization[0]
    else:
        return DEFAULT_QUALITY
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


def main(callback=None, loaded_model=None, argv=None, images=None, return_bytes=False):
    """Pass callback when processing multiple files with a curve model.

    callback (callable): call when finishing a file, pass input_path (str), True, info_str
    If error occurs: pass input_path (str), False, error message (str) to proceed or
    return an error message to abort.
    loaded_model (pt or tf model as returned by inference.get_model)
    argv (list): command line args to use instead of sys.argv[1:]
    images (list): images as BytesIO objects
    """
    argv = argv or sys.argv[1:]
    parser = get_parser()
    arg = parser.parse_args(argv)

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
    export_version = None
    if arg.export:
        supported_exports = {'darktable', }
        export_version = arg.export[1] if len(arg.export) > 1 else None
        arg.export = arg.export[0]
        if arg.export not in supported_exports:
            return f'Error: invalid export {arg.export}, must be one of {supported_exports}'
    if not (arg.outsuffix and arg.outsuffix.endswith('.xmp')):
        import shutil
        exiftool_path = arg.exiftool or shutil.which('exiftool')
    if arg.model:
        for fn in arg.model:
            if not Path(fn).exists():
                return f'Error: Specified model file could not be found: {fn}'
    else:
        if arg.outsuffix and arg.outsuffix.endswith('.xmp'):
            return 'Error: cannot export curves to darktable XMP without a model'
        if arg.export == 'darktable':
            print('Warning: ignoring option --export darktable, no model specified')
    if arg.input_icc_profile and arg.input_icc_profile.lower() != 'srgb':
        # TODO: implement more built-in ICC profiles and handle version consistent with
        # output_icc_profile
        icc_file = Path(arg.input_icc_profile)
        if not icc_file.exists():
            return f'Error: file not found: {icc_file}'
        input_icc_profile = get_icc_profile(icc_file, exiftool_path)
        print(f"DEBUG: input ICC profile from {icc_file}: {input_icc_profile['description']}")
    else:
        input_icc_profile = None  # read from each input file
    if arg.output_icc_profile:
        if arg.output_icc_profile.lower() == 'srgb':
            icc_version = input_icc_profile['version'] if input_icc_profile else '2.0.0'
            print(f"DEBUG: Creating sRGB version {icc_version} as target profile")
            output_icc_profile = get_srgb_profile(icc_version, exiftool_path)
        else:
            icc_file = Path(arg.output_icc_profile)
            if not icc_file.exists():
                return f'Error: file not found: {icc_file}'
            output_icc_profile = get_icc_profile(icc_file, exiftool_path)
            print(f"DEBUG: output ICC profile from {icc_file}: {output_icc_profile['description']}")
    else:
        output_icc_profile = None  # use input_rgb_profile
    model_space = arg.model_space.lower()
    if model_space not in {'none', 'trc', 'srgb', 'gamma'}:
        return f'Error: invalid model space {arg.model_space}'

    # Input file names
    path = Path(arg.folder)
    if not path.exists():
        return f'Error: folder "{path}" does not exist.'
    pre = arg.prefix
    if pre.startswith(('.', '/')):
        return f'Error: unsecure prefix "{pre}", use --folder to specify the path'
    suf = arg.suffix

    if images is not None:
        if not isinstance(images, list):
            return f'Error: images must be a list, got {type(images)}'
        fns = [Path(fn) for fn in arg.files]
        if len(fns) != len(images):
            return f'autolevels called with {len(fns)} file names but {len(images)} images passed.'
    elif arg.fstring:
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

        if loaded_model is not None:
            model = loaded_model  # model passed to main
        elif len(arg.model) == 1:
            model = get_model(arg.model[0])
        else:
            model = get_ensemble(arg.model)

    # Process input files
    for i, fn in enumerate(fns):
        # Skip non-existing
        if (images is None) and not fn.exists():
            print(f"Error: {fn} not found - skipping")
            if callback is not None:
                callback(str(fn), False, 'not found - skipping')
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
            if '.' not in suf:
                suf += ext  # add missing extension
            out_fn = (outdir or fn.parent) / f'{stem}{suf}'
        # TODO: check out_fn exists, add option -f to overwrite

        # Open image with PIL for metadata
        try:
            # This should work with file names, BytesIO, and streamlit.UploadedFile objects
            pil_img = Image.open(fn if (images is None) else BytesIO(images[i]))
        except Exception as e:
            print(f'Error: skipping {fn}, {e}')
            if callback is not None:
                callback(str(fn), False, 'unsupported or corrupt image format - skipping')
            if len(fns) == 1:
                # Return if this was the only file to process and it failed.
                if 'pil_img' in locals():
                    pil_img.close()
                return f'Unsupported or corrupt image format: {fn}'
            continue
        out_format = get_out_format(out_fn, pil_img)

        # Open/decode image with cv2 to get actual pixel array
        if images is not None:
            # Unpack streamlit.UploadedFile for cv2 with unknown bit-depth
            array = imread_unicode(fn, src_bytes=BytesIO(images[i]).read())
        else:
            array = imread_unicode(fn)
        maxvalue = 65535 if array.dtype == np.dtype('uint16') else 255

        # Get ICC profile from input file if no ICC file was provided, fallback: sRGB
        if input_icc_profile is None:
            print("DEBUG: Trying to load ICC profile from input file...")
        input_icc_profile = input_icc_profile or get_icc_profile(fn, exiftool_path)
        if input_icc_profile is None:
            input_icc_profile = get_srgb_profile(
                version=output_icc_profile['version'] if output_icc_profile else '2.0.0',
                exiftool_path=exiftool_path)
            print(f"Assuming sRGB: {input_icc_profile['description']}")

        # Check conditions for 48-bit output
        out_48bit = all((array.dtype == np.dtype('uint16'),
                         out_format in {'PNG', 'TIFF'}))

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
            L = grayscale(array)  # float32
            array = blend(array, L, saturation)  # float32

            # Convert from float32 to uint16
            array = (array * (65535 / 255) if maxvalue == 255 else array).round().clip(0, 65535).astype('uint16')
            maxvalue = 65535

        if arg.model:
            # Simulate: just test inference on first image
            if arg.simulate and fn != fns[0]:
                print(f'{fn} -> {out_fn}')
                pil_img.close()
                continue

            resized = cv2.resize(array, (384, 384)[::-1])  # uint16 or uint8
            if input_icc_profile['description'].startswith('sRGB') or model_space == 'none':
                free_curve = model(resized)
            elif model_space == 'srgb':
                icc_version = input_icc_profile['version'] if input_icc_profile else '2.0.0'
                srgb_profile = get_srgb_profile(icc_version, exiftool_path)
                resized = resized.astype(np.float32) / maxvalue
                resized = profile_to_profile(resized, input_icc_profile, srgb_profile)
                resized = (resized * maxvalue).round().astype(np.uint8 if maxvalue == 255 else np.uint16)
                #Image.fromarray(resized if maxvalue == 255 else (resized * (255 / 65535)).astype(np.uint8)).save('debug.png')
                #print("DEBUG: wrote model input to debug.png")
                free_curve = model(resized)
                free_curve = convert_curve_profile(free_curve, input_icc_profile, srgb_profile)
            elif model_space == 'trc':
                if 'srgb_trcs' not in globals():
                    srgb_trcs = None
                resized, srgb_trcs = convert_to_srgb(resized, input_icc_profile, exiftool_path, srgb_trcs)
                #Image.fromarray(resized if maxvalue == 255 else (resized * (255 / 65535)).astype(np.uint8)).save('debug.png')
                #print("DEBUG: wrote model input to debug.png")
                free_curve = model(resized)
                free_curve = convert_curve(free_curve, input_icc_profile, srgb_trcs)
            elif model_space == 'gamma':
                # Infer gamma of input_color_space
                input_gamma = infer_gamma(input_icc_profile, exiftool_path)
                resized = resized.astype(np.float32) / maxvalue
                resized = np.power(resized, input_gamma / 2.2)
                resized = (resized * maxvalue).round().astype(np.uint8 if maxvalue == 255 else np.uint16)
                #Image.fromarray(resized if maxvalue == 255 else (resized * (255 / 65535)).astype(np.uint8)).save('debug.png')
                #print("DEBUG: wrote model input to debug.png")
                free_curve = model(resized)
                free_curve = convert_curve_gamma(free_curve, input_gamma)

            # Export curves to supported programs
            if arg.export == 'darktable' or out_fn.suffix.endswith('.xmp'):
                from .export import append_rgbcurve_history_item

                # darktable xmp suffixes: .png.xmp (default), _01.png.xmp (via outsuffix, outfstring)
                if out_fn.suffix.endswith('.xmp'):
                    xmp_file = out_fn
                    skip_image_output = True
                else:
                    xmp_file = fn.with_name(fn.name + '.xmp')
                    skip_image_output = False

                try:
                    # if xmp exists, icc will be ignored, otherwise consider for xmp generation
                    append_rgbcurve_history_item(xmp_file, free_curve, pil_img,
                                                 icc=arg.input_icc_profile,
                                                 export_version=export_version)
                except Exception as e:
                    print(f'Error: failed generating {xmp_file}, skipping darktable export.')
                    print(e)  # DEBUG
                    if skip_image_output:
                        pil_img.close()
                        continue

                if skip_image_output:
                    print(f'{fn} -> {xmp_file}')
                    pil_img.close()
                    continue

            #Image.fromarray(array if maxvalue == 255 else (array * (255 / 65535)).astype(np.uint8)).save('debug_before_free_curve_map.png')
            array = free_curve_map_image(array, free_curve)  # float32, range (0, 1)
            #Image.fromarray((array * 255).astype(np.uint8)).save('debug_after_free_curve_map.png')

            if arg.simulate:
                print(f'{fn} -> {out_fn}')
                pil_img.close()
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

            # Set black point to min(target_black, blackpoint)
            target_black = np.minimum(target_black, blackpoint)

            # Set white point to max(target_white, whitepoint) or preserve it.
            if KEEP_WHITE and (target_white is None):
                whitepoint = np.array([255, 255, 255])
            target_white = whitepoint if target_white is None else np.maximum(target_white, whitepoint)

            # Simulate: just print black and white points
            if arg.simulate:
                print(f'{fn} -> {out_fn} (black point: {blackpoint} -> {target_black.round().astype("int")}, '
                      f'white point: {whitepoint} -> {target_white.round().astype("int")})')
                pil_img.close()
                continue

            # Make target black/white points gamma-agnostic
            black = 255 * np.power(target_black / 255, 1 / gamma)
            white = 255 * np.power(target_white / 255, 1 / gamma)

            shift = (blackpoint - black) * white / (white - black) if (white > black).all() else np.zeros_like(whitepoint)
            stretch_factor = white / np.clip(whitepoint - shift, 0, None)  # inf handled gracefully

            array = array.astype(np.float32)
            shift = shift.astype(np.float32)
            stretch_factor = stretch_factor.astype(np.float32)
            array = (array - shift * (maxvalue / 255)) * stretch_factor
            if (shift < 0).any():
                # small gamma results in a low black point => upper limit for target_black!
                channels = [name for name, s in zip('RGB', shift) if s < 0]
                print(f'{fn} WARNING: lower black point or increase gamma for channel(s)', *channels)

            array = np.clip(array / maxvalue, 0, 1)

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

        # Convert to output color space
        if output_icc_profile is None:
            output_icc_profile = input_icc_profile
        elif output_icc_profile['description'] != input_icc_profile['description']:
            print(f"Converting image from {input_icc_profile['description']} to {output_icc_profile['description']}")
            array = profile_to_profile(array, input_icc_profile, output_icc_profile, arg.rendering_intent,
                                       lut_interpolation=arg.lut_interpolation)

        kwargs = {}  # TODO: allow user to set save options, fill kwargs accordingly for cv2/PIL
        if out_48bit:
            # Quantize to 16-bit
            array = (array * 65535).round().clip(0, 65535).astype('uint16')
            imwrite_unicode(out_fn, array, default_ext='.png')
        else:
            # Quantize to 8-bit, continue with PIL Image
            array = (array * 255).round().clip(0, 255).astype('uint8')
            img = Image.fromarray(array)
            del array

            # Merge with alpha (RGBA images only)
            if img_alpha is not None:
                img = Image.merge('RGBA', [*img.split(), img_alpha])

            # Configure save options
            if out_format in {'JPEG'}:
                # Preserve JPEG quality
                from PIL import JpegImagePlugin
                if pil_img.format in {'JPEG'}:
                    kwargs['subsampling'] = JpegImagePlugin.get_sampling(pil_img)
                    kwargs['qtables'] = pil_img.quantization
                elif pil_img.format == 'TIFF' and pil_img.info.get('compression', '') == 'jpeg':
                    # qtables from TIFF are for RGB, not YCbCr, just keep quality-level
                    from autolevels.tiff_processor import extract_jpeg_info_from_tiff
                    qtables, subsampling = extract_jpeg_info_from_tiff(pil_img)
                    quality = estimate_jpeg_quality(qtables)
                    kwargs['quality'] = 44 + round((quality - 24) * (100 - 44) / (100 - 24))
                    kwargs['subsampling'] = subsampling
                else:
                    kwargs['quality'] = DEFAULT_QUALITY
                    kwargs['subsampling'] = 2 if pil_img.format in {'AVIF'} else 0
            elif out_format in {'TIFF'} and pil_img.format == 'JPEG':
                # Try to preserve visual input image JPEG quality (not file size).
                # TIFF files with JPEG-compression are larger because
                # - encode RGB, not YCbCr -> no effective quantization/Huffman coding (2.1×)
                # - 4:2:0 chroma subsampling (1.2×)
                # - stripes/overhead (< 1%)
                # Lower quality factor a bit (empirically) to match YCbCr compression quality
                kwargs['compression'] = 'jpeg'
                jpeg_quality = estimate_jpeg_quality(pil_img)
                jpeg_quality = max(24, 24 + round((jpeg_quality - 44) * (100 - 24) / (100 - 44)))
                kwargs['quality'] = jpeg_quality
            elif out_format == 'TIFF':
                # Keep input image compression if available (TIFF -> TIFF)
                kwargs['compression'] = pil_img.info.get('compression', 'tiff_adobe_deflate')
                if kwargs['compression'] == 'jpeg':
                    from autolevels.tiff_processor import extract_jpeg_info_from_tiff
                    qtables, subsampling = extract_jpeg_info_from_tiff(pil_img)
                    jpeg_quality = estimate_jpeg_quality(qtables)
                    kwargs['quality'] = jpeg_quality

            # Make reproducible, leave CLI args in JPEG comment
            if getattr(arg, 'cli_params', None):
                cli_params = arg.cli_params
            else:
                cli_params = purge_cli_params(argv, fn)
            comment = make_comment(pil_img, __version__, cli_params)

            if return_bytes:
                out_fn = BytesIO()
                comment = ''
                kwargs['format'] = out_format

            try:
                # Let PIL derive file format from extension
                img.save(out_fn, comment=comment, optimize=True, **kwargs)
            except ValueError as e:
                # If that fails, save in original format
                print(f"{e}, saving in {pil_img.format}.")
                img.save(out_fn, format=pil_img.format, comment=comment, optimize=True, **kwargs)

            if return_bytes:
                # No EXIF is needed for previews
                pil_img.close()
                return out_fn.getvalue()

        if images is None and not arg.skip_metadata:
            transfer_metadata(fn, out_fn, out_format, kwargs, exiftool_path,
                              input_icc_profile, output_icc_profile)

        # Clean up
        pil_img.close()

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
