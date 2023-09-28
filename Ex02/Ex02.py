import colorsys
import math

import cv2
import numpy as np


def calculate_hue(color):
    r, g, b = color
    h, _, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h


def show_color_table(color_table, string):
    num_squares_per_row = math.ceil(math.sqrt(len(color_table)))
    color_table.sort(key=calculate_hue)

    # Determine the size of each square and the number of squares
    square_size = 25
    image_size = square_size * num_squares_per_row

    # Create a blank square image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Draw each color as a square in the image
    for i, color in enumerate(color_table):
        row = i // num_squares_per_row
        col = i % num_squares_per_row
        start_x = col * square_size
        end_x = (col + 1) * square_size
        start_y = row * square_size
        end_y = (row + 1) * square_size
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, -1)

    cv2.imshow('Color Table ' + string, image)
    cv2.imwrite('Output_Images/Lena_' + string + '_Color_Table.png', image)


def compute_image_with_color_table(image, color_table, string):
    # create a blank image
    new_image = np.zeros((height, width, channels), np.uint8)

    # Read each pixel in image
    for i in range(height):
        for j in range(width):
            closest_color = min(color_table,
                                key=lambda x: abs(x[0] - image[i, j, 0]) + abs(x[1] - image[i, j, 1]) + abs(
                                    x[2] - image[i, j, 2]))
            new_image[i, j] = closest_color

    # Display image with 216 colors
    cv2.imshow('Image with ' + string + ' Color Table', new_image)
    cv2.imwrite('Output_Images/Lena_' + string + '.png', new_image)


def compute_universal_color_table():
    universal_color_table = []
    # Best use: 6x6x6 = 216 colors
    n_colors = 216
    step = 36

    for i in range(0, n_colors, step):
        for j in range(0, n_colors, step):
            for k in range(0, n_colors, step):
                universal_color_table.append((i, j, k))

    return universal_color_table


def compute_adaptive_color_table():
    adaptive_color_table = []
    # group pixels by 256 (16x16)
    for i in range(0, height, 16):
        for j in range(0, width, 16):
            # mean value of each group
            mean = np.mean(img[i:i + 16, j:j + 16], axis=(0, 1))
            adaptive_color_table.append(mean)

    return adaptive_color_table


if __name__ == "__main__":
    # Read image in color
    img = cv2.imread('Images/Lena.png', 1)

    # Display original image
    cv2.imshow('Original Image', img)
    cv2.imwrite('Output_Images/Lena_Original_Image.png', img)

    print("Image shape:", img.shape)
    height, width, channels = img.shape

    universal_color_table = compute_universal_color_table()
    show_color_table(universal_color_table, "universal")
    compute_image_with_color_table(img, universal_color_table, "universal")

    adaptive_color_table = compute_adaptive_color_table()
    show_color_table(adaptive_color_table, "adaptive")
    compute_image_with_color_table(img, adaptive_color_table, "adaptive")

    cv2.waitKey(0)
