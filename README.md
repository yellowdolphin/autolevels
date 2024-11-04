# Autolevels

AutoLevels is a program for batch-processing images to fix most common issues in a semi- for fully-automated fashion.


## Purpose

When you encounter images scanned from analog film or poorly processed by automatic corrections, such as "backlight" or simply underexposed, you will find black/white points that are too high/low. Even worse, they can differ by channel and produce a weird glow in dark areas or color cast on the entire image. A typical quick solution is offered by "autolevel" or contrast enhancing features, which set black and white points to zero and 255, respectively. This again might overshoot the problem, the result may appear unnatural, in particular if the original image has a low contrast.

AutoLevels helps you fix these issues by letting you choose sensible black/white points for a batch of images. It detects low-contrast images and treats them differently. Along the way, you can remove a constant color cast, change gamma and saturation. If your color cast or bad camera settings require a more complex curve correction, AutoLevels has you covered: try the AI-based free curve correction, which finds the optimal RGB curves for each image.


## Features

- Adjust black point, white point, gamma, and saturation
- Smooth/Histogram/Perceptive black/white point sampling
- Automatically detect low-contrast images and apply sensible corrections
- Fully automated curve correction
- Flexible definition of input/output files (glob pattern, pre/suffix, python f-string)
- Preserves JPEG properties (quality, EXIF)
- Open source, free software (GPLv3)


## Installation

If you have python 3.8 or later installed on your computer (Linux, MacOS), open a shell and execute

```bash
pip install git+https://github.com/yellowdolphin/autolevels.git
```

This will install also the following requirements if not found:
- numpy
- pillow
- piexif

If you want to use the fully automated curve correction feature, two additional steps are needed:
1. Install [PyTorch](https://pytorch.org/)
2. Download a [curve inference model]()

Now, you should be good to go:
```bash
autolevels --model {your_downloaded_model_file} -- example.jpg
```