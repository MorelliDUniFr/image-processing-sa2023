# Assigment 3

- author: Davide Morelli
- github link: https://github.com/MorelliDUniFr/image-processing-sa2023/tree/master/Ex03

## Description of the algorithms

### Algorithm 1: HSL to RGB
This algorithm does the same thing as we saw in class. We take the HSL values for each pixel in the image, and we compute the RGB values for each pixel, using the formulas we saw in class.

### Algorithm 2: RGB to HSL
This algorithm does the same thing as we saw in class. We take the RGB values for each pixel, normalize them between 0 and 1,
and then we compute the HSL values for each pixel, using the formulas we saw in class.

### Algorithm 3: Histogram equalization
At first we need to compute the histogram of the original greyscale image. This is done using the histogram method developed by numpy,
then we compute the cumulative distribution function (CDF) of the histogram. We normalize the CDF to have values between 0 and 255,
and then we use the normalized CDF to compute the new values of the pixels in the image.

## Response to question 3. c)
The image equalized using the RGB one, is more realistic looking to our eye. It has lost all the reddish colors, and it looks like a normal image.
In the other case, equalizing from the HSL image, we obtain a similar image to the original one, but with a augmented contrast on the colors.
