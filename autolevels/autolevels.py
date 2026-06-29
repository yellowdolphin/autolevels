#!/usr/bin/env python3
__version__ = '1.4.0'

from pathlib import Path
from argparse import ArgumentParser
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import re
from time import perf_counter

import numpy as np
from PIL import Image, ImageFilter, JpegImagePlugin

import imageio.v3 as iio
from exiftool import ExifToolHelper
from exiftool.exceptions import ExifToolExecuteError

from autolevels.icc.icc import (get_icc_profile, get_invertible_intents, profile_to_profile,
                                infer_gamma, convert_curve_gamma, convert_curve_profile)


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
                                'Options: sRGB (default), gamma, none (stay in input space).'))
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


def next_free_path(folder: str | Path, stem: str, suf: str) -> Path:
    """
    Find the next free path in `folder` by incrementing the number in the suffix.

    "?" in `suf` are replaced by a number of digits, and the next free number is returned.

    Args:
        folder (str | Path): Folder to search for the next free path
        stem (str): Stem of the path
        suf (str): Suffix of the path

    Returns:
        Path: The next free path
    """
    folder = Path(folder)

    m = re.search(r"\?+", suf)
    if not m:
        path = folder / f"{stem}{suf}"
        return path

    q_start, q_end = m.span()
    width = q_end - q_start

    prefix = suf[:q_start]
    suffix = suf[q_end:]

    # Use * instead of ? so the glob catches numbers longer than `width` digits
    glob_suf = f"{prefix}*{suffix}"

    pattern = re.compile(
        rf"^{re.escape(stem)}"
        rf"{re.escape(prefix)}"
        rf"(\d{{{width},}})"          # at least `width` digits
        rf"{re.escape(suffix)}$"
    )

    max_num = -1

    for path in folder.glob(f"{stem}{glob_suf}"):
        m = pattern.match(path.name)
        if m:
            max_num = max(max_num, int(m.group(1)))

    next_num = max_num + 1

    return folder / (
        f"{stem}{prefix}{next_num:0{width}d}{suffix}"
    )


def detect_image_format(path: str | Path) -> str | None:
    with open(path, "rb") as f:
        header = f.read(32)

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"

    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"

    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "GIF"

    if header.startswith(b"BM"):
        return "BMP"

    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "TIFF"

    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "WEBP"

    if header.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
        return "JPEG2000"

    return None


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
    SUPPORTED_OUT_FORMATS = {  # at least ICC write support
        'AI', 'AIT', 'ARQ', 'ARW', 'AVIF', 'CR2', 'CS1', 'DCP', 'DNG', 'EPS', 'EPSF', 'PS',
        'ERF', 'EXV', 'FFF', 'FLIF', 'GIF', 'GPR', 'HDP', 'WDP', 'JXR', 'HEIC', 'HEIF', 'HIF',
        'INSP', 'JPEG', 'JPG', 'JPE', 'MEF', 'MIE', 'MOS', 'MPO', 'MRW', 'NEF', 'NRW', 'ORF',
        'ORI', 'PEF', 'PNG', 'JNG', 'MNG', 'PSD', 'PSB', 'PSDT', 'RAF', 'RAW', 'RW2', 'RWL',
        'SR2', 'SRW', 'THM', 'TIFF', 'TIF', 'WEBP', 'X3F'}
    t0 = perf_counter()
    if not exiftool_path or not Path(exiftool_path).exists():
        print(f"exiftool not found, metadata is not preserved in {out_fn}.")
        return

    if out_format not in SUPPORTED_OUT_FORMATS:
        # Skipping exiftool entirely saves ~0.3 s
        print(f"no metadata support for {out_format}, metadata is not preserved in {out_fn}")
        return

    # Embed ICC profile when known
    if output_icc_profile and output_icc_profile.path is not None:
        icc_path = output_icc_profile.path
        if icc_path.suffix.lower() in {'.icc', '.icm'}:
            icc_args = [f'-icc_profile<={icc_path}']  # from ICC file
            #print(f"embedding profile from {icc_path}")
        elif icc_path == fn:
            icc_args = ['-icc_profile<icc_profile']  # embedded, from input image
            #print(f"copying profile from input file")
        else:
            icc_args = ['-tagsfromfile', f'{icc_path}', '-icc_profile<icc_profile']
            #print(f"embedding profile from {icc_path}")
    elif (output_icc_profile and input_icc_profile
          and output_icc_profile.name == input_icc_profile.name):
        # copy ICC profile from input image
        icc_args = ['-icc_profile<icc_profile']
        #print("copying profile from input file")
    elif input_icc_profile:
        print(f"Warning: no source file for sRGB v2 ICC profile (input image: {input_icc_profile.name})")
        icc_args = []  # should only happen if user-requested built-in profile with missing ICC file
    else:
        icc_args = []  # don't embed sRGB v2 by default when input image has no ICC profile, neither

    if out_format == 'TIFF' and kwargs.get('compression', '') == 'jpeg':
        # Injecting metadata to TIFF with JPEG compression is unsafe
        from autolevels.tiff_processor import exiftool_safe_transfer
        result = exiftool_safe_transfer(fn, out_fn, icc_args, exiftool_path)
        if result is False:
            print(f"exiftool could not transfer all metadata to {out_fn}.")
        return

    exiftool_args = ['-overwrite_original', '-tagsfromfile', f'{fn}', '-all:all']
    exiftool_args.extend(icc_args)
    exiftool_args.append(f'{out_fn}')

    with ExifToolHelper(executable=exiftool_path) as et:
        try:
            et.execute(*[a.encode() for a in exiftool_args])

        except ExifToolExecuteError as e:
            print(f"exiftool error: {e}")
            print(f"    {e.stdout}")
            print(f"    {e.stderr}")

        except Exception as e:
            print(f"exiftool error: {e}")

    t1 = perf_counter()
    print(f"Wall transfer_metadata: {(t1-t0)*1000:.3f} ms")


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

    # PIL requires ndim 2 for gray images (mode L)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    # Convert to PIL.Image
    if array.dtype == np.dtype('uint16'):
        img = Image.fromarray((array // 256).astype('uint8'))
    elif array.dtype.kind == 'f':
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


def get_out_format(filename):
    """Infer format from filename extension or use input format"""
    ext = Path(filename).suffix.lower()
    pil_extensions = Image.registered_extensions()
    return pil_extensions.get(ext)


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


def make_comment(pil_info, version, cli_params):
    """Save program version and CLI parameters in JPEG comment or EXIF"""

    comments = []

    # Keep existing comments
    if 'comment' in pil_info:
        try:
            comments.append(pil_info['comment'].decode())
        except UnicodeDecodeError:
            pass  # drop non-text comments

    comments.append(f'autolevels {version}, params: {cli_params}')

    return '\n'.join(comments)


def main(callback=None, loaded_model=None, argv=None, images=None, return_bytes=False):
    """Pass callback when processing multiple files with a curve model.

    Args:
        callback (callable): call when finishing a file, pass
            (input_path (str), True, info_str) after succesful iteration,
            (input_path (str), False, error_message (str)) if iteration fails.
            It returns: True to continue or False to abort (return None or 'user abort')
        loaded_model (PyTorch, tensorflow, or onnx model as returned by inference.get_model)
        argv (list): command line args to use instead of sys.argv[1:]
        images (list): images as BytesIO objects

    Returns:
        bytes (image) if return_bytes, else
        None (normal termination) or str (error message, if global error occurs)
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
    if arg.input_icc_profile:
        # Set global profile for all input images
        profile = get_icc_profile(arg.input_icc_profile, add_tags=True)
        if profile and hasattr(profile, 'name'):
            arg.input_icc_profile = profile
        else:
            return f'Error: "{arg.input_icc_profile}" not found and not a built-in profile'
        print(f"DEBUG: global source ICC profile: {arg.input_icc_profile.name}")
    if arg.output_icc_profile:
        # Set global profile for all output images
        profile = get_icc_profile(arg.output_icc_profile, add_tags=True)
        if profile and hasattr(profile, 'name'):
            arg.output_icc_profile = profile
        else:
            return f'Error: "{arg.output_icc_profile}" not found and not a built-in profile'
        print(f"DEBUG: global target ICC profile: {arg.output_icc_profile.name}")
    model_space = arg.model_space.lower()
    if model_space not in {'none', 'srgb', 'gamma'}:
        return f'Error: invalid model space {arg.model_space}'
    if arg.rendering_intent.startswith('relative'):
        arg.rendering_intent = 'relative_colorimetric'
    elif arg.rendering_intent.startswith('absolute'):
        arg.rendering_intent = 'absolute_colorimetric'

    # Input file names
    path = Path(arg.folder)
    if not path.is_dir():
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
            assert fn.is_file(), f'File not found: {fn}'
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
        # Input file does not exist, skip or return
        if (images is None) and not fn.is_file():
            if callable(callback):
                return '{fn} not found'
            else:
                print(f"Error: {fn} not found - skipping")
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
            if '?' in suf:
                out_fn = next_free_path(outdir or fn.parent, stem, suf)
            else:
                out_fn = (outdir or fn.parent) / f'{stem}{suf}'

        # Get metadata from file header
        try:
            # This works with file names, BytesIO, and streamlit.UploadedFile objects
            with Image.open(fn if (images is None) else BytesIO(images[i])) as img:
                in_format = img.format
                out_format = get_out_format(out_fn) or img.format or 'JPEG'
                pil_info = img.info
                image_alpha = img.getchannel('A') if img.mode in {'RGBA', 'LA'} else None
                exif = img.getexif()
                if in_format == 'JPEG':
                    jpeg_subsampling = JpegImagePlugin.get_sampling(img)
                    jpeg_quantization = img.quantization
                elif in_format == 'TIFF' and pil_info.get('compression', '') == 'jpeg':
                    from autolevels.tiff_processor import extract_jpeg_info_from_tiff
                    jpeg_quantization, jpeg_subsampling = extract_jpeg_info_from_tiff(img)

        except Exception as e:
            # PIL cannot handle XYZ, Lab. ImageIO provides no useful metadata.
            print(f'PIL could not open {fn} for metadata: {e}')
            in_format = detect_image_format(fn)
            out_format = get_out_format(out_fn) or in_format or 'JPEG'
            image_alpha = None  # determine from array or input_icc_profile later
            pil_info = {}

        # Open/decode image with ImageIO to get actual pixel array
        if images is not None:
            array = iio.imread(BytesIO(images[i]))
        else:
            try:
                if fn.suffix.lower() == '.png':
                    array = iio.imread(fn, plugin='PNG-FI')  # load RGB16 correctly
                    #array = iio.imread(fn, plugin='opencv', flags=-1)[:, :, ::-1]  # load RGB16 correctly
                else:
                    array = iio.imread(fn)
            except ValueError as e:
                if pil_info:
                    print(f"ImageIO: {e}, using PIL instead.")
                    with Image.open(fn if (images is None) else BytesIO(images[i])) as img:
                        array = np.asarray(img)
                else:
                    # Image cannot be decoded, return or continue with next image
                    print('unsupported or corrupt image format - skipping')
                    if callable(callback) or len(fns) == 1:
                        # Return if this was the only file to process and it failed.
                        return f'Unsupported or corrupt image format: {fn}'
                    else:
                        continue

        maxvalue = 65535 if array.dtype == np.dtype('uint16') else 255 if array.dtype == np.dtype('uint8') else 1
        if array.ndim == 3 and array.shape[2] in {2, 4}:
            image_alpha = array[:, :, -1]

        # Get ICC profile from input file if no global ICC file was specified, fallback: sRGB
        input_icc_profile = arg.input_icc_profile or get_icc_profile(fn, add_tags=True) or get_icc_profile('sRGB', add_tags=True)
        output_icc_profile = arg.output_icc_profile or input_icc_profile
        print(f"DEBUG: Input  ICC profile: {input_icc_profile.name}")
        print(f"DEBUG: Output ICC profile: {output_icc_profile.name}")

        # Check conditions for 48-bit output
        out_48bit = all((array.dtype == np.dtype('uint16'),
                         out_format in {'PNG', 'TIFF'}))

        # Handle grayscale images
        if array.ndim == 2:
            array = array[..., None]  # add 3rd dim if missing
        is_grayscale = True if array.shape[2] < 3 else False

        if is_grayscale and (arg.saturation != 1):
            print(f'Warning: "{fn}" is gray scale image, ignoring saturation options.')
            saturation = 1
        else:
            saturation = arg.saturation

        # Handle transparency in image modes RGBA, LA
        if image_alpha is not None:
            transparency = image_alpha.min() < maxvalue
            # Paste transparent images on white canvas
            if transparency:
                if array.dtype == np.uint8:
                    # Fast 8-bit PIL Version
                    pil_img = Image.fromarray(array)
                    if array.shape[-1] == 4:
                        canvas = Image.new('RGB', pil_img.size, (255, 255, 255))
                    else:
                        canvas = Image.new('L', pil_img.size, 255)
                    canvas.paste(pil_img, mask=pil_img)
                    array = np.array(canvas)
                else:
                    # Universial float32 Version
                    dtype = array.dtype
                    opacity = (image_alpha.astype(np.float32) / maxvalue)[..., None]
                    array = (array[..., 0] if array.shape[-1] == 2 else array[..., :3]).astype(np.float32) / maxvalue
                    array = (array - 1) * opacity + 1
                    array = (array.clip(0, 1) * maxvalue).round().astype(dtype)
            else:
                # Discard alpha channel if fully opaque
                image_alpha = None
                array = array[..., :3] if array.shape[-1] == 4 else array[..., :1]
                print(f"output format does not support transparency {array.shape}")

            # Also discard alpha channel if not supported by output format
            if out_format in {'JPEG'}:
                image_alpha = None

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
                continue

            # Resize and floatify model input before conversion to sRGB
            #print(f"DEBUG: model_input before resize: {array.dtype} {array.shape} {array.mean(axis=(0, 1))}")
            model_input = array.astype(np.float32) / maxvalue
            resized = []
            for channel in model_input.transpose(2, 0, 1):
                pil_channel = Image.fromarray(channel, mode='F')
                resized_channel = pil_channel.resize((384, 384), resample=Image.NEAREST)
                resized.append(np.array(resized_channel))
                #print(f"input channel mean: {channel.mean():8.4f}")
                #print(f"resized mean:       {resized[-1].mean():8.4f}")
            model_input = np.stack(resized, axis=2)
            #print(f"DEBUG: model_input from after resize: {model_input.dtype} {model_input.shape} {model_input.mean(axis=(0, 1))}")

            # Convert grayscale to RGB
            if is_grayscale:
                model_input = np.tile(model_input, (1, 1, 3))
            #print(f"DEBUG: model_input: {model_input.shape} {model_input.dtype} {model_input.min()} {model_input.mean()} {model_input.max()}")

            invertible_intents = get_invertible_intents(input_icc_profile, is_grayscale)
            #print(f"DEBUG: invertible_intents: {invertible_intents}")
            if 'sRGB' in input_icc_profile.name or model_space == 'none':
                free_curve = model(model_input)
                #Image.fromarray((model_input.clip(0, 1) * 255).astype(np.uint8)).save(f'{arg.outdir}/debug.png')
                #print(f"DEBUG: wrote model input to {arg.outdir}/debug.png")
            elif model_space == 'srgb' and invertible_intents:
                srgb_profile = get_icc_profile('sRGB')
                rendering_intent = (
                    'relative_colorimetric' if 'relative_colorimetric' in invertible_intents else
                    invertible_intents.pop())
                print(f"Converting model input {input_icc_profile.name} -> {srgb_profile.name} ({rendering_intent})")
                model_input = profile_to_profile(model_input, input_icc_profile, srgb_profile, rendering_intent)
                Image.fromarray((model_input.clip(0, 1) * 255).astype(np.uint8)).save(f'{arg.outdir}/debug.png')
                print(f"DEBUG: wrote model input to {arg.outdir}/debug.png")
                free_curve = model(model_input)
                free_curve = convert_curve_profile(free_curve, input_icc_profile, srgb_profile, rendering_intent)
            elif model_space == 'gamma' or not invertible_intents:
                print("DEBUG: adapting gamma of model input")
                # Infer gamma of input_color_space
                input_gamma = infer_gamma(input_icc_profile)
                model_input = np.power(model_input, input_gamma / 2.2)
                Image.fromarray((model_input.clip(0, 1) * 255).astype(np.uint8)).save('debug.png')
                print("DEBUG: wrote model input to debug.png")
                free_curve = model(model_input)
                free_curve = convert_curve_gamma(free_curve, input_gamma)
            else:
                raise ValueError(f"unknown model space adaptation: {model_space}")

            # Keep gray images gray
            if is_grayscale:
                free_curve = np.tile(free_curve.reshape(1, 3, 256).mean(axis=1, keepdims=True), (1, 1, 3))

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
                    if arg.simulate:
                        skip_image_output = True
                    else:
                        # icc_path is ignored if xmp_file exists, otherwise considered for xmp generation
                        icc_path = arg.input_icc_profile.path if arg.input_icc_profile else None
                        append_rgbcurve_history_item(xmp_file, free_curve, fn, exif,
                                                     icc_profile=input_icc_profile,
                                                     export_version=export_version)
                except Exception as e:
                    print(f'Error: failed generating {xmp_file}, skipping darktable export.')
                    print(e)  # DEBUG
                    if skip_image_output:
                        continue

                if skip_image_output:
                    print(f'{fn} -> {xmp_file}')
                    continue

            print(f"array before free_curve_map: {array.shape}")
            array = free_curve_map_image(array, free_curve)  # float32, range (0, 1)
            print(f"array after  free_curve_map: {array.shape}")

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
        if output_icc_profile.name != input_icc_profile.name:
            print(f"Converting image from {input_icc_profile.name} to {output_icc_profile.name}")
            array = profile_to_profile(array, input_icc_profile, output_icc_profile, arg.rendering_intent)

        kwargs = {}  # TODO: allow user to set save options, fill kwargs accordingly for ImageIO/PIL
        if out_48bit:
            if not ('XYZ' in output_icc_profile.name or 'Lab' in output_icc_profile.name):
                # Quantize to 16-bit
                #print(array.dtype, array.shape, array.min(), array.max())
                array = (array * 65535).round().clip(0, 65535).astype('uint16')
            if out_format == 'PNG':
                # ImageIO by default cannot save 48-bit PNG (employs PIL for PNG). 2 Options:
                # (A) opencv plugin: 50 MB extra package size for cv2, loads/saves reasonably fast
                # iio.imwrite(out_fn, array, plugin='opencv')  # 295 ms

                # (B) freeimage plugin: 8x slower (compression=6) or 10% larger file (compression=1)
                import imageio
                try:
                    imageio.plugins.freeimage.download()  # once to install the lib
                    iio.imwrite(out_fn, array, plugin='PNG-FI', compression=6)  # 995 ms
                except Exception as e:
                    print(f"ImageIO: {e}")
                    if callable(callback):
                        return f'{e}'
                    else:
                        continue
            elif out_format == 'TIFF':
                try:
                    iio.imwrite(out_fn, array, plugin='tifffile', compression="zlib", compressionargs={'level': 3}, predictor=True)
                except Exception as e:
                    print(f"ImageIO: {e}")
                    if callable(callback):
                        return f'{e}'
                    else:
                        continue
            else:
                try:
                    iio.imwrite(out_fn, array)
                except TypeError as e:
                    print(f"ImageIO: {e}")
                    if callable(callback):
                        return f'{e}'
                    else:
                        continue
        else:
            # Quantize to 8-bit, continue with PIL Image
            array = (array * 255).round().clip(0, 255).astype('uint8')
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]  # PIL requires ndim 2 for gray images (mode L)
            img = Image.fromarray(array)
            del array

            # Merge with alpha (LA, RGBA images only)
            if image_alpha is not None:
                image_alpha = Image.fromarray(image_alpha, mode='L')
                img = Image.merge('LA' if img.mode == 'L' else 'RGBA', [*img.split(), image_alpha])

            # Configure save options
            if out_format in {'JPEG'}:
                # Preserve JPEG quality
                if in_format in {'JPEG'}:
                    kwargs['subsampling'] = jpeg_subsampling
                    kwargs['qtables'] = jpeg_quantization
                elif in_format == 'TIFF' and pil_info.get('compression', '') == 'jpeg':
                    # qtables from TIFF are for RGB, not YCbCr, just keep quality-level
                    quality = estimate_jpeg_quality(jpeg_quantization)
                    kwargs['quality'] = 44 + round((quality - 24) * (100 - 44) / (100 - 24))
                    kwargs['subsampling'] = jpeg_subsampling
                else:
                    kwargs['quality'] = DEFAULT_QUALITY
                    kwargs['subsampling'] = 2 if in_format in {'AVIF'} else 0
            elif out_format in {'TIFF'} and in_format == 'JPEG':
                # Try to preserve visual input image JPEG quality (not file size).
                # TIFF files with JPEG-compression are larger because
                # - encode RGB, not YCbCr -> no effective quantization/Huffman coding (2.1×)
                # - 4:2:0 chroma subsampling (1.2×)
                # - stripes/overhead (< 1%)
                # Lower quality factor a bit (empirically) to match YCbCr compression quality
                kwargs['compression'] = 'jpeg'
                jpeg_quality = estimate_jpeg_quality(jpeg_quantization)
                jpeg_quality = max(24, 24 + round((jpeg_quality - 44) * (100 - 24) / (100 - 44)))
                kwargs['quality'] = jpeg_quality
            elif out_format == 'TIFF':
                # Keep input image compression if available (TIFF -> TIFF)
                kwargs['compression'] = pil_info.get('compression', 'tiff_adobe_deflate')
                if kwargs['compression'] == 'jpeg':
                    jpeg_quality = estimate_jpeg_quality(jpeg_quantization)
                    kwargs['quality'] = jpeg_quality

            # Make reproducible, leave CLI args in JPEG comment
            if getattr(arg, 'cli_params', None):
                cli_params = arg.cli_params
            else:
                cli_params = purge_cli_params(argv, fn)
            comment = make_comment(pil_info, __version__, cli_params)

            if return_bytes:
                out_fn = BytesIO()
                comment = ''
                kwargs['format'] = out_format

            try:
                # Let PIL derive file format from extension
                img.save(out_fn, comment=comment, optimize=True, **kwargs)
            except ValueError as e:
                # Attempted alternatives:
                # - ImageIO: could extend supported formats, but may save in different format (TIFF)
                #   without warning -> don't use for now.
                # - tifffile for TIFF with JPEG compression: has issues (bad preview and other tags),
                #   requires imagecodecs, an EXIF parser (piexif), and code for composing the extratags list.
                # - Drop previous behavior saving in input format if suffix is not recognized:
                #   img.save(out_fn, format=in_format, comment=comment, optimize=True, **kwargs)
                print(f"PIL: {e}")
                if callable(callback):
                    return f'{e}'
                else:
                    continue

            if return_bytes:
                # Return image for previews and streamlit, no metadata or infos are needed
                return out_fn.getvalue()

        if images is None and not arg.skip_metadata:
            transfer_metadata(fn, out_fn, out_format, kwargs, exiftool_path,
                              input_icc_profile, output_icc_profile)

        # Logging
        infos = [f'{fn} -> {out_fn}']
        if not arg.model and (blackpoint != target_black).any():
            high = 'high ' if (blackpoint > max_black).any() else ''
            infos.append(f'{high}black point: {blackpoint} -> {target_black.round().astype("int")}')
        if not arg.model and (whitepoint != target_white).any():
            low = 'low ' if (whitepoint < min_white).any() else ''
            infos.append(f'{low}white point: {whitepoint} -> {target_white.round().astype("int")}')
        print(', '.join(infos))

        # Call callback for user-abort feature
        if callable(callback) and callback(str(fn), True, infos) is False:
            return 'user abort'


if __name__ == '__main__':
    main()
