# Assigment 2

- author: Davide Morelli
- github link: https://github.com/MorelliDUniFr/image-processing-sa2023/tree/master/Ex02

## Description of the algorithms

### Algorithm 1: universal color table
We need to chose firstly what number gives us the best range of colors: turns out that the number is 216 (6 values for each RGB value).
We can compute a fixed step for each color channel and form our 216 colors.

In order to find the best correspondence of our color to the original one, we can compute the euclidean distance between the two colors and chose the one with the minimum distance.
With that, we have an image that has a reduced amount of colors, but maintaining a good quality.

### Algorithm 2: adaptive color table
If we want to improve our result, we can compute a color table for each image.
We have a maximum of 256 colors, and the image (in this case at least) has a size of 256x256 pixels. So we can reduce the pixel quality to 16x16 "new pixels" and find the mean color of each of them.
With that, we have a color table that is specific for each image, and we can use it to reduce the colors of the image.

## Results
If we compare the results of the two images, we can clearly see that the second one (adaptive color table) is better than the first one (universal color table).