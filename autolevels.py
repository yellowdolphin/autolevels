from pathlib import Path
from PIL import Image
from PIL import ImageFilter
import numpy as np
from shutil import copy2
from argparse import ArgumentParser
from glob import glob


parser = ArgumentParser(description='Set proper blackpoint for each image channel')
parser.add_argument('--blackpoint', nargs='+', default=14, help="target for soft blackpoint")
parser.add_argument('--pixel', default=0.005, help="percentage of pixel darker than blackpoint")
parser.add_argument('--mode', default='hist', help='sample mode: "smooth", "smoother", or "hist"')
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
        array = np.array(smoothened)  # HWC
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


for fn in fns:
    out_fn = fn.parent / (fn.stem + '_al' + fn.suffix)

    img = Image.open(fn)
    blackpoint = get_blackpoint(img, sample_mode, thr_pixel)

    if (blackpoint < thr_black).all():
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
    #print("normalized array:", array.min(axis=(0, 1)), array.max(axis=(0, 1)))
    array = array.clip(0, 255)
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

