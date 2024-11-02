#!/home/wanko/miniconda3/bin/python
__version__ = '0.1.1'

from pathlib import Path
from shutil import copy2
from PIL import Image, ImageFilter, ImageEnhance
import piexif
from argparse import ArgumentParser
from time import perf_counter
import sys

import numpy as np


KEEP_WHITE = False  # keep white instead of whitepoint if no whitepoint is specified
BENCHMARK = False
DEFAULT_QUALITY = 75
REPRODUCIBLE = {'blackpoint', 'whitepoint', 'blackclip', 'whiteclip', 'maxblack', 'minwhite', 'max_blackshift',
                'max_whiteshift', 'mode', 'gamma', 'model', 'saturation', 'saturation_first', 'saturation_before_gamma'}

parser = ArgumentParser(description='Example: autolevels --blackpoint 10 --whitepoint 255 --gamma 1.2 input.jpg')

points = parser.add_argument_group("Black and white point correction")
points.add_argument('--blackpoint', nargs='+', default=14, type=int,
                                    help=("Target black point, one L or three RGB values, range 0...255 (default: 14). "
                                          "The image black point is measured per channel. If it is higher than the target "
                                          "value, it will be lowered, otherwise kept unchanged."))
points.add_argument('--whitepoint', nargs='+', default=None, type=int,
                                    help=("Target white point, one L or three RGB values, range 0...255 (default: keep). "
                                          "The image white point is not changed by default, and will be increased to "
                                          "the target value, if specified."))

clipping = parser.add_argument_group("Clip shadows and highlights")
# adobe auto color defaults are between 0.001 and 0.005
clipping.add_argument('--blackclip', nargs='+', default=0.002, type=float,
                                     help=("Percentage of pixels darker than black point (shadows clipped). "
                                           "Due to noise and sharpening, the mathematical black point "
                                           "can be lower than the perceived one. To mitigate this, the image black "
                                           "point can be lowered until a certain percentage of the pixels is darker "
                                           "than the set target value for the black point. The default of 0.002 "
                                           "ignores the darkest 0.2 percent of the pixels when calculating the "
                                           "current black point."))
clipping.add_argument('--whiteclip', nargs='+', default=0.001, type=float,
                                     help=("Percentage of pixels brighter than white point (highlights clipped). "
                                           "The default of 0.001 ingores the brightest 0.1 percent of the pixels when "
                                           "calculating the current white point."))

limit = parser.add_argument_group("If the image is low in contrast, limit the correction of shadows and highlights")
limit.add_argument('--maxblack', nargs='+', default=75, type=int,
                                 help=("Max black point (L or RGB) value(s) for which the full correction is applied "
                                       "(default: 75)."))
limit.add_argument('--minwhite', nargs='+', default=240, type=int,
                                 help=("Min white point (L or RGB) value(s) for which the full correction is applied "
                                       "(default: 240)."))
limit.add_argument('--max-blackshift', nargs='+', default=[27, 22, 28], type=int,
                                       help=("Max black-point shift if MAXBLACK is exceeded (default: [27, 22, 28]). "
                                             "If the current black point is higher than MAXBLACK, this sets an upper "
                                             "bound for the applied shift."))
limit.add_argument('--max-whiteshift', nargs='+', default=20, type=int,
                                       help=("Max white-point shift if MINWHITE is not achieved (default: 20). If the "
                                             "white point is lower than MINWHITE, this sets an upper bound for the "
                                             "applied shift."))

bp_mode = parser.add_argument_group("Mode for determining the black and white points")
bp_mode.add_argument('--mode', default='hist', choices=['smooth', 'smoother', 'hist', 'perceptive'],
                               help=('Black/white point sample mode: "smooth", "smoother", "hist" (default), or '
                                     '"perceptive". "smooth" takes the pixel min/max values from a copy of the image '
                                     'smoothened with a 3x3 envelope to compensate for noise/sharpen effects. '
                                     '"smoother" does the same with a 5x5 envelope. '
                                     '"hist" calculates the values at which a fraction of BLACKCLIP (WHITECLIP) '
                                     'pixels is darker (brighter) than the black (white) point, respectively. '
                                     '"perceptive" does the same with a weighted histogram, which is slower but can '
                                     'improve the blackpoint of images with color cast. Use --simulate to show the '
                                     'measured black and white points before and after processing.'))

curve = parser.add_argument_group("Curve corrections")
curve.add_argument('--gamma', nargs='+', type=float, default=[1.0],
                              help=('Gamma correction with inverse gamma (larger is brighter), one L or three RGB values '
                                    '(default: 1.0).'))
curve.add_argument('--model', nargs='+', action="store", help="Model file(s) for free-curve correction")

sat = parser.add_argument_group("Saturation")
sat.add_argument('--saturation', default=1, type=float,
                                 help=("A value of 0 produces a gray image, a value larger than 1 increases saturation "
                                       "(default: 1.0)."))
sat.add_argument('--saturation-first', action='store_true', help="Adjust saturation before anything else")
sat.add_argument('--saturation-before-gamma', action='store_true', help="Adjust saturation before gamma (deprecated)")

file_location = parser.add_argument_group("File locations")
file_location.add_argument('--folder', default='.', help="Path to input images")
file_location.add_argument('--prefix', default='', help="Common prefix of all input file names")
file_location.add_argument('--suffix', default='',
                                       help=('Common suffix (including file extension) of all input file names'))
file_location.add_argument('--fstring', default=None,
                                        help=('Expand input file names using a Python f-string. '
                                              'Example: --fstring f"IMG_{x:04d}.jpg" -- 3 4 5  expands to  '
                                              'IMG_0003.jpg IMG_0004.jpg IMG_0005.jpg'))
file_location.add_argument('--outdir', '--outfolder', default=None, help='Write output files here (default: original folder)')
file_location.add_argument('--outsuffix', default=None, type=str,
                                          help=('Suffix (including file extension) used in output file names. '
                                                'Default: append "_al" to input file name (before file extension). '
                                                'If both SUFFIX and OUTSUFFIX are specified, OUTSUFFIX replaces '
                                                'SUFFIX in the output file name.'))
file_location.add_argument('--outprefix', default=None, type=str,
                                          help=('Prefix used in output file names. Default: none or same as input file name. '
                                                'If both PREFIX and OUTPREFIX are specified, OUTPREFIX replaces SUFFIX in '
                                                'the output file name.'))
file_location.add_argument('--outfstring', default=None, type=str,
                                           help=('If input file names are expanded using a Python f-string FSTRING, an '
                                                 'alternative f-string can be specified here for output files. Otherwise, '
                                                 'this option will be ignored.'))

parser.add_argument('--simulate', '--sandbox', action='store_true',
                                               help="Dry run: only read and process, skip file output")
parser.add_argument('--reproduce', default='', help=("Read CLI options from metadata of specified image REPRODUCE. "
                                                     "The latter must be the output of a compatible program version. "
                                                     "Example: autolevels --reproduce processed_image.jpg "
                                                     "other_images/*.jpg"))
parser.add_argument('--version', action='store_true', help="Print version information and exit")

parser.add_argument('files', nargs='*', action="store", help=('Input files to process. Example: scans/IMG_*.jpg'))


def extract_arg(filename, parser):
    """Extract args from a previous autolevel output file"""

    old_namespace = parser.parse_args()

    filename = Path(filename)
    assert filename.exists(), f'No file {filename}'

    img = Image.open(filename)

    cli_params = ''

    # parse JPEG comment
    if hasattr(img, 'info') and 'comment' in img.info:
        comment = img.info['comment'].decode()
        comment = comment.split('\n')[-1]  # multiline: read only last
        if 'autolevels ' in comment:
            version = comment.split('autolevels ')[1].split(',')[0]
            if version != __version__:
                print(f"WARNING: autolevels version changed: {version} -> {__version__}")
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
        assert name in current_arg, f'WARNING: foreign parameter {name}, update reproducible_params!'
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
    assert len(s) > 1, f'The f-string is improperly formatted or missing quotes'
    s = s[1:] if (s[0] == 'f') else s  # remove leading "f"
    s = '"' + s + '"' if (s[0] not in {'"', "'"}) else s  # add missing quotes
    s = 'f' + s if ('{' in s) else s  # add leading "f" if required

    # Checks on the f-string: starts with "f", quotes, len limits
    if not all([len(s) >= 5, len(s) < 1000, s.endswith(s[1]), s.count('{') <= 1, s.count('}') <= 1]):
        raise ValueError("The f-string is improperly formatted or missing quotes")

    # Find all variable patterns in the f-string, allowing whitespace around the variable name
    matches = re.findall(r'\{\s*(\w+)\s*(:[^}]*)?\}', s)
    if len(matches) == 0:
        raise ValueError(f"No valid variable symbol found in the f-string")
    elif len(matches) > 1:
        raise ValueError("The f-string contains more than one variable symbol")

    # Extract the single variable name and its specifier
    var_name, specifier = matches[0]
    var_name = var_name.strip()  # Remove any surrounding whitespace
    specifier = specifier if specifier else ""

    # Check if the variable name is a valid Python identifier
    if not var_name.isidentifier():
        raise ValueError(f"Invalid variable name '{var_name}' in the f-string")

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


def get_blackpoint_whitepoint(img, mode, pixel_black, pixel_white):
    # 3x3 or 5x5 envelope
    SMOOTH = ImageFilter.SMOOTH_MORE if mode == 'smoother' else ImageFilter.SMOOTH

    if mode.startswith('smooth'):
        img = img.filter(SMOOTH)
        array = np.array(img, dtype=np.float32)  # HWC
        return array.min(axis=(0, 1)), array.max(axis=(0, 1))

    elif mode.startswith('perceptive'):
        if BENCHMARK:
            ts = [0, 0, 0, 0]
            t0 = perf_counter()
        assert img.mode == 'RGB', f'image mode "{img.mode}" not supported by perceptive sampling mode'
        R, G, B = (np.array(channel, dtype=np.float32) for channel in img.split())  # 0.37 s
        if BENCHMARK: ts[0], t0 = ts[0] + perf_counter() - t0, perf_counter()
        L = np.array(img.convert(mode='L'), dtype=np.float32)  # 0.06 s
        L_bp = L.min()
        L = (L - L_bp) * (255 / (255 - L_bp)) + 0.5
        if BENCHMARK: ts[1], t0 = ts[1] + perf_counter() - t0, perf_counter()
        pixel_black = pixel_black if pixel_black.shape == (3,) else pixel_black.repeat(3)
        pixel_white = pixel_white if pixel_white.shape == (3,) else pixel_white.repeat(3)
        blackpoint, whitepoint = [], []

        for pix_black, pix_white, channel in zip(pixel_black, pixel_white, (R, G, B)):
            weight = np.where(channel >= L, 1, channel / L)  # 0.20 s
            if BENCHMARK: ts[2], t0 = ts[2] + perf_counter() - t0, perf_counter()
            #assert (channel >= 0).all()  # 0.10 s
            hist, _ = np.histogram(channel, bins=256, range=(0, 256), weights=weight)  # 0.49 s (0.58 with uint8)
            if BENCHMARK: ts[3], t0 = ts[3] + perf_counter() - t0, perf_counter()
            n_pixel = hist.sum()

            # the rest takes no time
            accsum = 0
            for x in range(256):
                accsum += hist[x]
                if accsum > n_pixel * pix_black:
                    break
            blackpoint.append(x)

            accsum = 0
            for x in range(255, -1, -1):
                accsum += hist[x]
                if accsum > n_pixel * pix_white:
                    break
            whitepoint.append(x)

        if BENCHMARK: 
            for i, t in enumerate(ts):
                print(f"timer {i}: {t}")

        return np.array(blackpoint), np.array(whitepoint)

    elif mode in {'hist', 'histogram'}:
        channels = img.split()
        pixel_black = pixel_black if pixel_black.shape == (3,) else pixel_black.repeat(len(channels))
        pixel_white = pixel_white if pixel_white.shape == (3,) else pixel_white.repeat(len(channels))
        
        n_pixel = img.height * img.width
        blackpoint, whitepoint = [], []

        for pix_black, pix_white, channel in zip(pixel_black, pixel_white, channels):
            hist = channel.histogram()

            accsum = 0
            for x in range(256):
                accsum += hist[x]
                if accsum > n_pixel * pix_black:
                    break
            blackpoint.append(x)

            accsum = 0
            for x in range(255, -1, -1):
                accsum += hist[x]
                if accsum > n_pixel * pix_white:
                    break
            whitepoint.append(x)

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


def blend(a, b, alpha=1.0):
    "Interpolate between arrays `a`and `b`"
    return a if (alpha == 1) else alpha * a + (1.0 - alpha) * b


def grayscale(rgb, mode='itu', keep_channels=False):
    "Convert RGB image (float array) to L"

    if mode == 'itu':
        # Rec. ITU-R BT.601-7 definition of luminance
        R, G, B = (rgb[:, :, c] for c in range(3))
        L = R * 0.299 + G * 0.587 + B * 0.114
    elif mode == 'mean':
        L = rgb.mean(axis=2)
    else:
        raise ValueError(f"mode {mode} not supported")

    return np.stack([L, L, L]) if keep_channels else L[:, :, None]


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
    "Save program version and CLI parameters in JPEG comment or EXIF"

    comments = []

    if hasattr(img, 'info') and 'comment' in img.info:
        comments.append(img.info['comment'].decode())

    comments.append(f'autolevels {version}, params: {cli_params}')

    return '\n'.join(comments)


if __name__ == '__main__':
    arg = parser.parse_args()

    if arg.version:
        print(f"AutoLevels version {__version__}")
        exit()

    if not arg.files:
        parser.print_usage()
        exit("No files specified")

    # Post-process arg
    if arg.reproduce:
        extracted_arg = extract_arg(arg.reproduce, parser)
        assert hasattr(extracted_arg, 'cli_params'), 'called extract_arg but arg has no cli_params'
        arg = merge_args(current_arg=arg, extracted_arg=extracted_arg)
        assert hasattr(arg, 'cli_params'), 'merge_args deleted cli_params'
        print(f"Reproducing {arg.reproduce} processing: {arg.cli_params}")
    blackclip = np.array(arg.blackclip, dtype=float)
    max_black = np.array(arg.maxblack, dtype=int)
    max_blackshift = np.array(arg.max_blackshift, dtype=int)
    whiteclip = np.array(arg.whiteclip, dtype=float)
    min_white = np.array(arg.minwhite, dtype=int)
    max_whiteshift = np.array(arg.max_whiteshift, dtype=int)
    sample_mode = arg.mode
    assert all(g > 0 for g in arg.gamma), f'invalid gamma {arg.gamma}, must be positive'
    gamma = 1 / np.array(arg.gamma, dtype=float)
    saturation = arg.saturation
    if arg.model:
        for fn in arg.model:
            assert Path(fn).exists(), f'Specified model file could not be found: {fn}'

    # Input file names
    path = Path(arg.folder)
    assert path.exists(), f'Folder "{path}" does not exist.'
    pre = arg.prefix
    assert not pre.startswith(('.', '/')), f'Unsecure prefix "{pre}", use --folder to specify the path'
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
                print(f'Skipping "{x}"'); continue
            try:
                parent, stem, ext = Path(x).parent, Path(x).stem, Path(x).suffix
                #glob_pattern = f'{parent}/{pre}{stem}{suf}{ext}'
                #print("glob pattern:", glob_pattern)
                #fns.extend(sorted(path.glob(glob_pattern)))

                # Skip glob expansion for now
                fns.append(path / parent / f'{pre}{stem}{suf}{ext}')
            except IndexError:
                print("path:", path)
                print("pattern:", x)
                raise

    assert fns, f'No matching files found in "{path}"'

    # Output file options
    outdir = Path(arg.outdir) if arg.outdir else None
    if outdir and not arg.simulate:
        outdir.mkdir(exist_ok=True)


    # TODO: implement batchwise inference on fns
    if arg.model:
        # Free-curve correction from predicted curve
        from inference import get_model, get_ensemble, free_curve_map_image

        if len(arg.model) == 1:
            model = get_model(arg.model[0])
        else:
            model = get_ensemble(arg.model)


    for i, fn in enumerate(fns):
        if arg.outfstring:
            out_fn = (outdir or fn.parent) / evaluate_fstring(arg.outfstring, arg.files[i])
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

        img = Image.open(fn)

        # Adjust saturation before anything else
        if (saturation != 1 and arg.saturation_first):
            array = np.array(img, dtype=np.float32)
            L = grayscale(array)
            array = blend(array, L, saturation)

        if arg.model:
            # Simulate: just test inference on first image
            if arg.simulate and fn != fns[0]:
                print(f"{fn} -> {out_fn}")
                continue

            # Resize image for model input
            if 'array' in globals():
                array_8 = array.clip(0, 255).astype('uint8')
                resized = Image.fromarray(array_8, mode=(img.mode or 'RGB')).resize(model.input_size)
                quantization_noise = array_8.astype(np.float32) - array  # range: (-1, 0]
                plus_one = np.clip(array + 1, 0, 255).astype('uint8')
            else:
                array_8 = np.array(img)
                resized = img.resize(model.input_size)
                quantization_noise = None

            free_curve = model(np.array(resized, dtype=np.float32))

            array = free_curve_map_image(array_8, free_curve)

            # Eliminate quantization noise
            if quantization_noise is not None:
                transformed_plus_one = free_curve_map_image(plus_one, free_curve)
                transformed_noise = (transformed_plus_one - array) * quantization_noise  # negative
                array -= transformed_noise

            if arg.simulate:
                print(f"{fn} -> {out_fn}")
                continue

        else:
            blackpoint, whitepoint = get_blackpoint_whitepoint(img, sample_mode, blackclip, whiteclip)

            # Set targets, limit shifts in black/whitepoint for low-contrast images
            target_black = np.array(arg.blackpoint, dtype=int)
            target_white = np.array(arg.whitepoint, dtype=int) if arg.whitepoint else None
            if (blackpoint > max_black).any():
                shift = max_blackshift * (255 - blackpoint) / (255 - max_blackshift)
                target_black = np.maximum(target_black, blackpoint - shift)
            if (whitepoint < min_white).any() and arg.whitepoint:
                if np.var(max_whiteshift) == 0:
                    # avoid clipping to preserve hue + saturation of white point
                    max_whiteshift = np.minimum(max_whiteshift, (target_white - whitepoint).min())
                shift = max_whiteshift * whitepoint / (255 - max_whiteshift)
                target_white = np.minimum(target_white, whitepoint + shift)

            # Set blackpoint to min(target_black, blackpoint).
            target_black = np.minimum(target_black, blackpoint)

            # Set whitepoint to max(target_white, whitepoint) or preserve it.
            if KEEP_WHITE and (target_white is None):
                whitepoint = np.array([255, 255, 255])
            target_white = whitepoint if target_white is None else np.maximum(target_white, whitepoint)

            # Simulate: just print black and white points
            if arg.simulate:
                print(f"{fn} -> {out_fn} (black point: {blackpoint} -> {np.uint8(target_black)},", 
                    f"whitepoint: {whitepoint} -> {np.uint8(target_white)})")
                continue

            # Make target black/white points gamma-agnostic
            black = 255 * np.power(target_black / 255, 1 / gamma)
            white = 255 * np.power(target_white / 255, 1 / gamma)

            shift = (blackpoint - black) * white / (white - black)
            stretch_factor = white / (whitepoint - shift)
            if 'array' not in globals():
                array = np.array(img, dtype=np.float32)
            array = (array - shift) * stretch_factor
            #print(f"black: {black}, white: {white}")
            #print(f"shift: {shift}, stretch_factor: {stretch_factor}, min: {array.min(axis=(0, 1))}, max: {array.max(axis=(0, 1))}")
            if (shift < 0).any():
                # small gamma results in a low black point => upper limit for target_black!
                channels = [name for name, s in zip('RGB', shift) if s < 0]
                print(f"{fn} WARNING: lower black point or increase gamma for channel(s)", *channels)

        # Adjust saturation before gamma (deprecated)
        if (saturation != 1 and arg.saturation_before_gamma and not arg.saturation_first):
            L = grayscale(array)
            array = blend(array, L, saturation)

        # Gamma correction
        if (gamma != 1).any():
            array = array.clip(0, None)
            array = 255 * np.power(array / 255, gamma)

        # Adjust saturation
        if (saturation != 1 and not (arg.saturation_before_gamma or arg.saturation_first)):
            L = grayscale(array)
            array = blend(array, L, saturation)

        array = array.clip(0, 255)

        new_img = Image.fromarray(np.uint8(array))
        del array  # remove from namespace before next fn iteration

        # Add attributes required to preserve JPEG quality
        for attr in 'format layer layers quantization'.split():
            value = getattr(img, attr, None)
            if value is not None:
                setattr(new_img, attr, value)

        # Add other attributes (obsolete?)
        for attr in 'info'.split():
            value = getattr(img, attr, None)
            if value is not None:
                setattr(new_img, attr, value)

        quality = 'keep' if (img.format in {'JPEG'}) else DEFAULT_QUALITY

        # Make reproducible, leave CLI args in JPEG comment
        if getattr(arg, 'cli_params', None):
            cli_params = arg.cli_params
        else:
            cli_params = purge_cli_params(sys.argv[1:], fn)
        comment = make_comment(img, __version__, cli_params)

        # Save JPEG, regardless of output file extension (TODO: handle other formats and their metadata)
        new_img.save(out_fn, format=img.format, comment=comment, optimize=True, quality=quality)

        # Neither PIL nor piexif correctly decode the (proprietary) MakerNotes IFD.
        # Hence, this is the only way to fully preserve the entire EXIF:
        if hasattr(img, 'info') and 'exif' in img.info:
            piexif.transplant(str(fn), str(out_fn))

        # Logging
        infos = [f'{fn} -> {out_fn}']
        if not arg.model and (blackpoint != target_black).any():
            high = 'high ' if (blackpoint > max_black).any() else ''
            infos.append(f'{high}black point {blackpoint} -> {np.uint8(target_black)}')
        if not arg.model and (whitepoint != target_white).any():
            low = 'low ' if (whitepoint < min_white).any() else ''
            infos.append(f'{low}white point {whitepoint} -> {np.uint8(target_white)}')
        print(', '.join(infos))
