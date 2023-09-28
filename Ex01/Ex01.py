import cv2
import numpy as np


def flip_horizontally(image):
    print("Flipping image horizontally...")

    flipped_image = np.zeros((height, width, channels), np.uint8)

    # flip the image horizontally
    for i in range(height):
        for j in range(width):
            flipped_image[i, j] = image[i, width - j - 1]

    print("... image flipped horizontally\n")

    return flipped_image


def flip_vertically(image):
    print("Flipping image vertically...")

    flipped_image = np.zeros((height, width, channels), np.uint8)

    # flip the image vertically
    for i in range(height):
        for j in range(width):
            flipped_image[i, j] = image[height - i - 1, j]

    print("... image flipped vertically\n")

    return flipped_image


def flip_horizontally_and_vertically(image):
    print("Flipping image horizontally and vertically...")

    flipped_image = np.zeros((height, width, channels), np.uint8)

    # flip the image horizontally and vertically
    for i in range(height):
        for j in range(width):
            flipped_image[i, j] = image[height - i - 1, width - j - 1]

    print("... image flipped horizontally and vertically\n")

    return flipped_image


if __name__ == "__main__":
    # Read image in color
    img = cv2.imread('Images/Ara.png', 1)

    # Display original image
    cv2.imshow('Original Image', img)

    print("Image shape:", img.shape)
    height, width, channels = img.shape

    h_image = flip_horizontally(img)
    v_image = flip_vertically(img)
    hv_image = flip_horizontally_and_vertically(img)

    # Display the images
    cv2.imshow('Horizontally Flipped Image', h_image)
    cv2.imshow('Vertically Flipped Image', v_image)
    cv2.imshow('Flipped Image', hv_image)
    cv2.waitKey(0)

    # Save the images to file
    cv2.imwrite('Output_Images/Horizontally_Flipped_Image.png', h_image)
    cv2.imwrite('Output_Images/Vertically_Flipped_Image.png', v_image)
    cv2.imwrite('Output_Images/Flipped_Image.png', hv_image)
    print("Images saved to file\n")

