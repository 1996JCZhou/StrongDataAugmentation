# Strong data augmentation methods for color images

Code reproduction for the paper [RandAugment](https://arxiv.org/abs/1909.13719v2).

Examples of strong data augmentations:
1. Apply affine transformation on the image keeping image center invariant.
2. Shear angle value in degrees between -180 to 180, clockwise direction. If a sequence is specified, the first value corresponds to a shear parallel to the x axis, while the second value corresponds to a shear parallel to the y axis.
3. Maximize contrast of an image by remapping its pixels per channel so that the lowest becomes black and the lightest becomes white.
4. Equalize the histogram of an image by applying a non-linear mapping to the input in order to create a uniform distribution of grayscale values in the output.
5. Solarize an RGB/grayscale image by inverting all pixel values above a predefined threshold.
6. Posterize an image by reducing the number of bits for each color channel.

## Requirements
- torch
- torchvision

## Examples
Original image
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/swan.jpg)

Image after stronger contrast changes
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_Contrast_Stronger.JGP)

Image after equalization changes
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_Equalize.JGP)

Image after rotation with 120 degree
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_Rotation120.JGP)

Image after adding blur
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_Sharpness_Addblur.JGP)

Image after increasing sharpness
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_Sharpness_Increasesharpness.JGP)

Image after schearing along the x-axis with 30 degree
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_ShearX30.JGP)

Image after solarization
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_Solarize.JGP)

Image after solarization
![image](https://github.com/1996JCZhou/StrongDataAugmentation/blob/master/Examples/Image_Solarize.JGP)
