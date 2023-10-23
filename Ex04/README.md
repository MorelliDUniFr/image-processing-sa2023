# Assigment 4

- author: Davide Morelli
- github link: https://github.com/MorelliDUniFr/image-processing-sa2023/tree/master/Ex04

## Description of the algorithms

### Explanation of input/output size
In order to have the same input and output size, the image is padded with zeros.

### Gradient filter
We apply the gradient filter to the image, using the following kernels:
```
Gx = [-1 0 1; -2 0 2; -1 0 1]
Gy = [-1 -2 -1; 0 0 0; 1 2 1]
```
The gradient filter is applied to the image, and the result is the magnitude of the two gradients.
So we can see the edges in the image.

### Laplacian of Gaussian filter
We apply the Laplacian of Gaussian filter to the image, using the following kernel:
```
LoG = [0 0 -1 0 0; 0 -1 -2 -1 0; -1 -2 16 -2 -1; 0 -1 -2 -1 0; 0 0 -1 0 0]
```
The Laplacian of Gaussian filter is applied to the image, and the result is the edges in the image.

### Response question 1. c)
We can clearly see that if the filter applied is bigger, the blurriness of the image is more noticeable.

### Response question 2. d)
The Laplacian of Gaussian filter is more effective in detecting edges, because it is more sensitive to edges.
We can clearly see that difference in the output images.

### Response question 3. c)
The maximum filter enhances bright details and objects in the image.
The minimum filter enhances dark details and objects in the image.
The max-min filter highlights edges and boundaries between objects or regions with different intensities.
