# AutoLevels

AutoLevels is a program for batch-processing images to fix common issues in a semi- or fully-automated fashion.

## Purpose

When you encounter images scanned from analog film or poorly processed by automatic corrections, such as "backlight" or simply underexposed, you will find black/white points that are too high/low. Even worse, they can differ by channel and produce a weird glow in dark areas or color cast on the entire image. Photo editing apps typically have features like "autolevel" or "enhance contrast" that could be used in batch-processing. They set black and white points to zero and 255, respectively. These might overshoot the problem, and the result may appear unnatural, particularly if the original image has low contrast. They are also not very suitable for fixing color problems.

AutoLevels helps you fix these issues by letting you choose sensible black/white points for a batch of images. It detects low-contrast images and treats them differently. Along the way, you can remove a constant color cast, change gamma and saturation. If your color cast or bad camera settings require a more complex curve correction, AutoLevels has you covered: try the AI-based free curve correction, which finds the optimal RGB curves for each image.

## Features

- Adjust black point, white point, gamma, and saturation
- Smooth/Histogram/Perceptive black/white point sampling
- Automatically detect low-contrast images and apply sensible corrections
- Fully automated curve correction
- Support for 16/48-bit images (PNG, TIFF)
- Apply ICC color profiles *after* corrections
- Flexible definition of input/output files (glob pattern, prefix/suffix, Python f-string)
- Preserves JPEG properties (quality, EXIF)
- Open source, free software (GPLv3)

## Installation

If you have Python 3.9 or later installed on a Linux or MacOS machine, open a shell and execute:

```bash
pip install autolevels
```

This will install the current stable release. To get the latest version from the github repository (requires git), use

```bash
pip install git+https://github.com/yellowdolphin/autolevels.git
```

This will also install the following requirements if not found:
- numpy
- pillow
- piexif
- opencv-python
- h5py

If you want to use the fully automated curve correction feature, two additional steps are needed:
1. Install [PyTorch](https://pytorch.org/).
2. Download a [Free Curve Inference model](https://www.kaggle.com/models/greendolphin/freecin) as tar.gz and extract it:
```bash
tar -xzvf freecin-pytorch-xcittiny-v1.tar.gz
```

Now, you should be good to go:
```bash
autolevels --model {your_downloaded_model_file} -- example.jpg
```

## Documentation

### Basic Usage

```bash
autolevels --blackpoint 10 --whitepoint 255 --gamma 1.1 -- example.jpg
```

This will process the file `example.jpg` and write the output to `example_al.jpg` using the default suffix "`_al`". You can change that with `--outsuffix <my suffix>` or specify an output folder with `--outdir`. See **Batch Processing** for more ways to define input and output file names.

Get a description of all options with:

```bash
autolevels -h
```

Safely try out settings before writing any output files with `--simulate` or `--sandbox`. You can check file names, black/white point correction, as images are read and processed, but not saved.
```bash
autolevels --simulate --blackpoint 10 5 0 --mode perceptive -- *.jpg
```

### Batch Processing

The power of batch processing lets you apply the same corrections to a batch of images with a common capture source, camera settings, lighting conditions, or any common issue that can be fixed in a semi-automated fashion. If you don't have the time or expertise to find the optimal parameters, you can leverage AI power using the `--model` option. This will correct each image with an individual RGB curve, fully automatic, correcting color casts, bad exposure, or white balance settings on the fly.

This leaves you with defining input and output files and paths. AutoLevels gives you three ways to do that.

1. **Folders and glob patterns**
```bash
autolevels --outdir processed -- scans/*.png IMG_00[0-3]?.jpg
```
Your shell will interpret these glob patterns and expand the file names to `scans/12.png scans/23.png IMG_0015.jpg ...` matching any existing files in the current directory. All output files are written to the specified folder `processed`.

If you are afraid the expanded list of file paths exceeds the shell limit for the length of a command, you can specify a folder for the input files and enter the glob pattern in quotes to escape shell expansion:
```bash
autolevels --folder ~/Pictures/scans -- "*.tif"
```

2. **Prefixes and Suffixes**
Often, your input file names will have a common folder, prefix, suffix, and some variable part in between. You may want to keep the variable part and change any of the fixed components, for example:
```bash
autolevels --folder orig --prefix scn --suffix .jpg --outfolder processed --outprefix img_ --outsuffix .jpg -- 1 2 3 4
```
This will search for input files `orig/scn1.jpg ...` and write output files `processed/img_1.jpg ...`

3. **Python f-strings**
An alternative way to define file names in AutoLevels is to use Python f-strings. Don't worry, no Python skills required, just look at this example:
```python
variable = 42
f"orig/scn_{variable:04d}.jpg"
```
This is proper Python code and evaluates to "orig/scn_0042.jpg". The part in curly brackets contains a variable name and (after the `:`) an optional format instruction for integer numbers `d`, which shall have leading `0`s to make `4` digits.

If you provide an f-string to the `--fstring` or `--outfstring` options (you can skip the "f" before the quotes), AutoLevels substitutes `variable` by any values given instead of file names:
```bash
autolevels --fstring "orig/scn{i}.jpg" --outfstring "processed/img_{i:04d}.jpg" -- 1 2 3
```
This will read files `orig/scn1.jpg ...` and write `processed/img_0001.jpg ...` to disk.
