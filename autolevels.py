#!/home/wanko/miniconda3/bin/python

from pathlib import Path
from shutil import copy2
from PIL import Image, ImageFilter, ImageEnhance
from argparse import ArgumentParser

import numpy as np


KEEP_WHITE = False  # keep white instead of whitepoint if no whitepoint is specified

parser = ArgumentParser(description='Set proper blackpoint for each image channel')
parser.add_argument('--blackpoint', nargs='+', default=14, type=int, 
                                    help="Target for soft blackpoint, 1 luminance or 3 RGB values, range 0...255, default 14")
parser.add_argument('--whitepoint', nargs='+', default=None, type=int, 
                                    help="Target for soft whitepoint, 1 luminance or 3 RGB values, range 0...255, default: keep")
parser.add_argument('--blackpixel', nargs='+', default=0.005, type=float, 
                                    help="Percentage of pixels darker than blackpoint")
parser.add_argument('--whitepixel', nargs='+', default=0.001, type=float,
                                    help="Percentage of pixels brighter than whitepoint")
parser.add_argument('--mode', default='hist', choices=['smooth', 'smoother', 'hist'], 
                              help='Blackpoint sample mode: "smooth", "smoother", or "hist"')
parser.add_argument('--gamma', nargs='+', type=float, default=[1.0], 
                               help='Gamma correction with inverse gamma (larger=brighter), 1 global or 3 RGB values')
parser.add_argument('--saturation', default=1, type=float)
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
parser.add_argument('files', nargs='*', action="store", help='File names to process (supports glob patterns)')

arg = parser.parse_args()

thr_black = np.array(arg.blackpoint, dtype=int)
black_pixel = np.array(arg.blackpixel, dtype=float)
thr_white = np.array(arg.whitepoint, dtype=int) if arg.whitepoint else None
white_pixel = np.array(arg.whitepixel, dtype=float)
sample_mode = arg.mode
assert all(g > 0 for g in arg.gamma), f'invalid gamma {arg.gamma}, must be positive'
gamma = 1 / np.array(arg.gamma, dtype=float)
saturation = arg.saturation

path = Path(arg.folder)
assert path.exists(), f'Folder "{path}" does not exist.'
if arg.files:
    fns = []
    for x in arg.files:
        fns.extend(sorted(path.glob(x)))
else:
    pre = arg.prefix
    indices = arg.indices
    assert indices, 'specify either file name(s) or indices!'
    magazines = arg.magazines or [arg.magazine] * len(indices)
    sep = arg.separator
    fns = [path / f'{pre}{m}{sep}{i}.jpg' for m, i in zip(magazines, indices)]

assert fns, f"No matching files found in {path}."

outdir = Path(arg.outdir) if arg.outdir else None
if outdir and not arg.simulate:
    outdir.mkdir(exist_ok=True)


def get_blackpoint_whitepoint(img, mode, pixel_black, pixel_white):
    # 3x3 or 5x5 envelope
    SMOOTH = ImageFilter.SMOOTH_MORE if mode == 'smoother' else ImageFilter.SMOOTH

    if mode.startswith('smooth'):
        img = img.filter(SMOOTH)
        array = np.array(img)  # HWC
        return array.min(axis=(0, 1)), array.max(axis=(0, 1))

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


for fn in fns:
    out_fn = (outdir or fn.parent) / (fn.stem + arg.outsuffix + fn.suffix)

    img = Image.open(fn)

    blackpoint, whitepoint = get_blackpoint_whitepoint(img, sample_mode, black_pixel, white_pixel)

    if (blackpoint < thr_black).all() and (whitepoint < (thr_white or 256)).all() and (gamma == 1).all():
        #copy2(fn, ou_fn)
        #print(f"{fn} -> {out_fn} (no change)")
        print(f"skipping {fn} (black and white points are OK)")
        continue

    # Set black point to min(thr_black, blackpoint).
    black = np.minimum(thr_black, blackpoint)

    # Set white point to max(thr_white, whitepoint) or preserve it.
    if KEEP_WHITE and (thr_white is None):
        whitepoint = np.array([255, 255, 255])
    white = whitepoint if thr_white is None else np.maximum(thr_white, whitepoint)

    # Simulate: just print black and white points
    if arg.simulate:
        print(f"{fn} -> {out_fn} (blackpoint: {blackpoint} -> {black}, whitepoint: {whitepoint} -> {white})")
        continue

    shift = (blackpoint - black) * white / (white - black)
    stretch_factor = white / (whitepoint - shift)
    array = np.array(img, dtype=np.float64)
    array = (array - shift) * stretch_factor

    # Adjust saturation
    if saturation != 1:
        L = grayscale(array)
        array = blend(array, L, saturation)

    array = array.clip(0, 255)
    
    # Gamma correction
    if (gamma != 1).any():
        array = array.clip(0, None)
        array = 255 * np.power(array / 255, gamma)

    img = Image.fromarray(np.uint8(array))
    #smoothened = img.filter(SMOOTH)
    #new_blackpoint = np.amin(smoothened, axis=(0, 1))
    #new_whitepoint = np.amax(smoothened, axis=(0, 1))
    #print("black:", black, "shift:", blackpoint - black, "stretch:", stretch_factor)
    #print(f"smoothened blackpoint {blackpoint} -> {new_blackpoint}")
    #print(f"smoothened whitepoint {whitepoint} -> {new_whitepoint}")
    print(f"{fn} -> {out_fn} (blackpoint: {blackpoint} -> {black}, whitepoint: {whitepoint} -> {white})")
    img.save(out_fn)
    #print()
