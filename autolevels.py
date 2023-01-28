from pathlib import Path
from shutil import copy2
from PIL import Image, ImageFilter, ImageEnhance
from argparse import ArgumentParser

import numpy as np


parser = ArgumentParser(description='Set proper blackpoint for each image channel')
parser.add_argument('--blackpoint', nargs='+', default=14, type=int, 
                                    help="Target for soft blackpoint, 1 luminance or 3 RGB values, range 0...255")
parser.add_argument('--pixel', default=0.005, type=float, help="percentage of pixel darker than blackpoint")
parser.add_argument('--mode', default='hist', choices=['smooth', 'smoother', 'hist'], 
                              help='sample mode: "smooth", "smoother", or "hist"')
parser.add_argument('--gamma', nargs='+', type=float, default=[1.0], 
                               help='Gamma correction with inverse gamma (larger=brighter), 1 global or 3 RGB values')
parser.add_argument('--saturation', default=1, type=float)
parser.add_argument('--folder', default='.')
parser.add_argument('--prefix', default='Magazin')
parser.add_argument('--magazine', default=1)
parser.add_argument('--magazines', nargs='+', default=None)
parser.add_argument('--indices', nargs='*')
parser.add_argument('--separator', default='.', help='magazine/index separator')
parser.add_argument('files', nargs='*', action="store", help='file names')

arg = parser.parse_args()

thr_black = np.array(arg.blackpoint, dtype=int)
thr_pixel = float(arg.pixel)
sample_mode = arg.mode
assert all(g > 0 for g in arg.gamma), f'invalid gamma {arg.gamma}, must be positive'
gamma = 1 / np.array(arg.gamma, dtype=float)
saturation = arg.saturation

path = Path(arg.folder)
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


def get_blackpoint(img, mode='smooth', thr_pixel=0.002):
    # 3x3 or 5x5 envelope
    SMOOTH = ImageFilter.SMOOTH_MORE if mode == 'smoother' else ImageFilter.SMOOTH

    if mode.startswith('smooth'):
        img = img.filter(SMOOTH)
        array = np.array(img)  # HWC
        return array.min(axis=(0, 1))

    elif mode.startswith('hist'):
        channels = img.split()
        n_pixel = img.height * img.width
        blackpoint = []

        for c in channels:
            hist = c.histogram()
            accsum = 0
            for x in range(256):
                accsum += hist[x]
                if accsum > n_pixel * thr_pixel:
                    break
            blackpoint.append(x)

        return np.array(blackpoint)


def blend(a, b, alpha=1.0):
    "Interpolate between arrays `a`and `b`"
    return a if (alpha == 1) else alpha * a + (1.0 - alpha) * b


def grayscale(rgb, mode='itu', keep_channels=False):
    "Convert RGB image (float array) to L"
    print(rgb.shape)
    R, G, B = (rgb[:, :, c] for c in range(3))

    if mode == 'itu':
        # Rec. ITU-R BT.601-7 definition of luminance
        L = R * 0.299 + G * 0.587 + B * 0.114
    else:
        raise ValueError(f"mode {mode} not supported")

    return np.stack([L, L, L]) if keep_channels else L[:, :, None]


for fn in fns:
    out_fn = fn.parent / (fn.stem + '_al' + fn.suffix)

    img = Image.open(fn)

    blackpoint = get_blackpoint(img, sample_mode, thr_pixel)

    if (blackpoint < thr_black).all() and (gamma == 1).all():
        #copy2(fn, ou_fn)
        #print(f"{fn} -> {out_fn} (no change)")
        print(f"skipping {fn} (blackpoint OK)")
        continue

    # Set black point to min(thr_black, blackpoint), preserve white point
    black = np.minimum(thr_black, blackpoint)
    whitepoint = np.array([255, 255, 255])
    shift = (blackpoint - black) * whitepoint / (whitepoint - black)
    stretch_factor = whitepoint / (whitepoint - shift)
    array = np.array(img, dtype=np.float64)
    array = (array - shift) * stretch_factor

    # Adjust saturation
    if saturation != 1:
        L = grayscale(array)
        array = blend(array, L, saturation)

    array = array.clip(0, 255)
    #print("normalized array:", array.min(axis=(0, 1)), array.max(axis=(0, 1)))
    
    # Gamma correction
    if (gamma != 1).any():
        array = 255 * np.power(array / 255, gamma)

    img = Image.fromarray(np.uint8(array))
    #smoothened = img.filter(SMOOTH)
    #new_blackpoint = np.amin(smoothened, axis=(0, 1))
    #new_whitepoint = np.amax(smoothened, axis=(0, 1))
    #print("black:", black, "shift:", blackpoint - black, "stretch:", stretch_factor)
    #print(f"smoothened blackpoint {blackpoint} -> {new_blackpoint}")
    #print(f"smoothened whitepoint {whitepoint} -> {new_whitepoint}")
    print(f"{fn} -> {out_fn} (blackpoint: {blackpoint} -> {thr_black})")
    img.save(out_fn)
    #print()
