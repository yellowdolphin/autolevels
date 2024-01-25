#!/home/wanko/miniconda3/bin/python
__version__ = '0.1.0'

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

parser = ArgumentParser(description='AutoLevels: batch image processing with common corrections')
parser.add_argument('--blackpoint', nargs='+', default=14, type=int, 
                                    help="Target black point, 1 luminance or 3 RGB values, range 0...255 (default: 14)")
parser.add_argument('--whitepoint', nargs='+', default=None, type=int, 
                                    help="Target white point, 1 luminance or 3 RGB values, range 0...255 (default: keep)")
parser.add_argument('--blackpixel', nargs='+', default=0.002, type=float, 
                                    help="Percentage of pixels darker than BLACKPOINT")
parser.add_argument('--whitepixel', nargs='+', default=0.001, type=float,
                                    help="Percentage of pixels brighter than WHITEPOINT")
parser.add_argument('--maxblack', nargs='+', default=75, type=int,
                                  help="Max allowed RGB value(s) for full black-point correction")
parser.add_argument('--minwhite', nargs='+', default=240, type=int,
                                  help="Min allowed RGB value(s) for full white-point correction")
parser.add_argument('--max-blackshift', nargs='+', default=[27, 22, 28], type=int,
                                        help="max black-point shift if MAXBLACK is exceeded")
parser.add_argument('--max-whiteshift', nargs='+', default=20, type=int,
                                        help="max white-point shift if MINWHITE is not achieved")
parser.add_argument('--mode', default='perceptive', choices=['smooth', 'smoother', 'hist', 'perceptive'], 
                              help='Black/white point sample mode: "smooth", "smoother", "hist", or "perceptive" (default)')
parser.add_argument('--gamma', nargs='+', type=float, default=[1.0], 
                               help='Gamma correction with inverse gamma (larger=brighter), 1 global or 3 RGB values')
parser.add_argument('--model', nargs='+', action="store", help="Model file(s) for free-curve correction")
parser.add_argument('--saturation', default=1, type=float)
parser.add_argument('--saturation-before-gamma', action='store_true', help="correct saturation before gamma (deprecated)")
parser.add_argument('--folder', default='.')
parser.add_argument('--prefix', default='Magazin')
parser.add_argument('--magazine', default=1)
parser.add_argument('--magazines', nargs='+', default=None)
parser.add_argument('--indices', nargs='*')
parser.add_argument('--separator', default='.', help='Magazine/index separator')
parser.add_argument('--outdir', default=None, help='Write output files here (default: original folder)')
parser.add_argument('--outsuffix', default='_al', 
                                   help='Append OUTSUFFIX to original file name (overwrite existing files)')
parser.add_argument('--simulate', '--sandbox', action='store_true')
parser.add_argument('--reproduce', default='', help="Specify earlier autolevels output, use same CLI options")
parser.add_argument('files', nargs='*', action="store", help='File names to process (supports glob patterns)')

arg = parser.parse_args()


def extract_args(filename, parser):
    "Extract args from a previous autolevel output file"

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
    return new_namespace


# post-process arg
if arg.reproduce:
    arg = extract_args(arg.reproduce, parser)
black_pixel = np.array(arg.blackpixel, dtype=float)
max_black = np.array(arg.maxblack, dtype=int)
max_blackshift = np.array(arg.max_blackshift, dtype=int)
white_pixel = np.array(arg.whitepixel, dtype=float)
min_white = np.array(arg.minwhite, dtype=int)
max_whiteshift = np.array(arg.max_whiteshift, dtype=int)
sample_mode = arg.mode
assert all(g > 0 for g in arg.gamma), f'invalid gamma {arg.gamma}, must be positive'
gamma = 1 / np.array(arg.gamma, dtype=float)
saturation = arg.saturation
if arg.model:
    for fn in arg.model:
        assert Path(fn).exists(), f'Specified model file could not be found: {fn}'

path = Path(arg.folder)
assert path.exists(), f'Folder "{path}" does not exist.'
if arg.files:
    fns = []
    for x in arg.files:
        if x in '. .. \ /'.split():
            print(f'Skipping "{x}"'); continue
        try:
            fns.extend(sorted(path.glob(x)))
        except IndexError:
            print("path:", path)
            print("pattern:", x)
            raise
else:
    pre = arg.prefix
    indices = arg.indices
    assert indices, 'specify either file name(s) or indices!'
    magazines = arg.magazines or [arg.magazine] * len(indices)
    sep = arg.separator
    fns = [path / f'{pre}{m}{sep}{i}.jpg' for m, i in zip(magazines, indices)]

assert fns, f'No matching files found in "{path}".'

outdir = Path(arg.outdir) if arg.outdir else None
if outdir and not arg.simulate:
    outdir.mkdir(exist_ok=True)


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
        L = np.array(img.convert(mode='L'), dtype=np.float32) + 1e-5  # 0.06 s
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

    elif mode.startswith('hist'):
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


def make_comment(img, version, cli_params):
    "Save program version and CLI parameters in JPEG comment or EXIF"

    comments = []

    if hasattr(img, 'info') and 'comment' in img.info:
        comments.append(img.info['comment'].decode())

    comments.append(f'autolevels {version}, params: {cli_params}')

    return '\n'.join(comments)


for fn in fns:
    out_fn = (outdir or fn.parent) / (fn.stem + arg.outsuffix + fn.suffix)

    img = Image.open(fn)

    # TODO: implement batchwise inference on fns
    if arg.model:
        # Free-curve correction from predicted curve
        from inference import get_model, get_ensemble, free_curve_map_image

        array = np.array(img.resize((384, 384)), dtype=np.float32)
        if len(arg.model) == 1:
            model = get_model(arg.model[0])
        else:
            model = get_ensemble(arg.model)
        free_curve = model(array)
        del model
        array = np.array(img)
        array = free_curve_map_image(array, free_curve)

    else:
        blackpoint, whitepoint = get_blackpoint_whitepoint(img, sample_mode, black_pixel, white_pixel)

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
            print(f"{fn} -> {out_fn} (blackpoint: {blackpoint} -> {np.uint8(target_black)},", 
                f"whitepoint: {whitepoint} -> {np.uint8(target_white)})")
            continue

        # Make target black/white points gamma-agnostic
        black = 255 * np.power(target_black / 255, 1 / gamma)
        white = 255 * np.power(target_white / 255, 1 / gamma)

        shift = (blackpoint - black) * white / (white - black)
        stretch_factor = white / (whitepoint - shift)
        array = np.array(img, dtype=np.float32)
        array = (array - shift) * stretch_factor
        #print(f"black: {black}, white: {white}")
        #print(f"shift: {shift}, stretch_factor: {stretch_factor}, min: {array.min(axis=(0, 1))}, max: {array.max(axis=(0, 1))}")
        if (shift < 0).any():
            # small gamma results in a low blackpoint => upper limit for target_black!
            channels = [name for name, s in zip('RGB', shift) if s < 0]
            print(f"{fn} WARNING: lower black point or increase gamma for channel(s)", *channels)

    # Adjust saturation (before gamma)
    if (saturation != 1 and arg.saturation_before_gamma):
        L = grayscale(array)
        array = blend(array, L, saturation)

    # Gamma correction
    if (gamma != 1).any():
        array = array.clip(0, None)
        array = 255 * np.power(array / 255, gamma)

    # Adjust saturation
    if (saturation != 1 and not arg.saturation_before_gamma):
        L = grayscale(array)
        array = blend(array, L, saturation)

    array = array.clip(0, 255)

    new_img = Image.fromarray(np.uint8(array))

    # Add attributes required to preserve JPEG quality
    for attr in 'format layer layers quantization'.split():
        value = getattr(img, attr, None)
        if value is not None:
            setattr(new_img, attr, value)

    # Add other attributes (obsolete?)
    for attr in 'mode info'.split():
        value = getattr(img, attr, None)
        if value is not None:
            setattr(new_img, attr, value)

    quality = 'keep' if (img.format in {'JPEG'}) else DEFAULT_QUALITY

    # Make reproducible, leave CLI args in JPEG comment
    cli_params = getattr(arg, 'cli_params', '') or ' '.join(sys.argv[1:]).split(' -- ')[0]
    comment = make_comment(img, __version__, cli_params)

    new_img.save(out_fn, format=img.format, comment=comment, optimize=True, quality=quality)

    # Neither PIL nor piexif correctly decode the (proprietary) MakerNotes IFD.
    # Hence, this is the only way to fully preserve the entire EXIF:
    piexif.transplant(str(fn), str(out_fn))

    # Logging
    infos = [f'{fn} -> {out_fn}']
    if not arg.model and (blackpoint != target_black).any():
        high = 'high ' if (blackpoint > max_black).any() else ''
        infos.append(f'{high}blackpoint {blackpoint} -> {np.uint8(target_black)}')
    if not arg.model and (whitepoint != target_white).any():
        low = 'low ' if (whitepoint < min_white).any() else ''
        infos.append(f'{low}whitepoint {whitepoint} -> {np.uint8(target_white)}')
    print(', '.join(infos))
