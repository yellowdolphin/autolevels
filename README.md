AutoLevels is a script for batch-processing images to fix most common errors in a smart and semi-automated fashion.

Features:
- Adjust black point, white point, gamma, and saturation
- Find perceptive rather than numeric black/white points
- Automatically detect low-contrast images and apply sensible corrections
- Flexible definition of input/output files
- Open source, free software (GPLv3)

When you encounter images scanned from analog films or poorly processed by automatic corrections, such as "backlight" or simply underexposed, you will find black/white points that are too high/low. Even worse, they can differ by channel and produce a weird glow in dark areas or color cast on the entire image. A typical quick solution is offered by "autolevel" or contrast enhancing features, which set black and white points to zero and 255, respectively. This again might overshoot the problem, the result may appear unnatural, in particular if the original image has a low contrast.

AutoLevels helps you fix these issues by letting you choose sensible black/white points for a batch of images. It detects low-contrast images and treats them differently. Along the way, you can remove a constant color cast, change gamma and saturation.

