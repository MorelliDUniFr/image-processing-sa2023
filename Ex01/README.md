# Assignement 1

author: Davide Morelli
github link: https://github.com/MorelliDUniFr/image-processing-sa2023/tree/master

## Flipping algorithm
In order to flip an image pixel by pixel, we fist need to read every pixel of the image. 
To copy every pixel in the correct position after the flip, we can create a blank image that has the size has the original one.
After that, depending on what flip we want to do (horizontal or vertical), we can copy the pixel in the correct position, going through the image from the last pixel to the first one.
In order to flip an image both horizontally and vertically, we can simply do both operations at the same time.