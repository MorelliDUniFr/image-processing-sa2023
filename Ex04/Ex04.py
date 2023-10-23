import cv2
import numpy as np


def convolutional_mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.int16)

    # Get the size of the image
    image_height, image_width, channels = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding required for the output
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Calculate the size of the padded image
    padded_height = image_height + 2 * pad_height
    padded_width = image_width + 2 * pad_width

    # Pad the image
    padded_image = np.zeros((padded_height, padded_width, channels), dtype=image.dtype)

    # Copy the image to the padded image
    padded_image[pad_height:pad_height + image_height, pad_width:pad_width + image_width, :] = image

    # Create the output image
    output_image = np.zeros((image_height, image_width, channels), dtype=image.dtype)

    # Apply the filter
    for row in range(image_height):
        for col in range(image_width):
            for channel in range(channels):
                output_image[row, col, channel] = np.sum(
                    np.multiply(kernel, padded_image[row:row + kernel_height, col:col + kernel_width, channel])) // (
                                                              kernel_height * kernel_width)

    return output_image


def laplacian_of_gaussian_filter(image, kernel):
    # Get the size of the image
    image_height, image_width, channels = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding required for the output
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Calculate the size of the padded image
    padded_height = image_height + 2 * pad_height
    padded_width = image_width + 2 * pad_width

    # Pad the image
    padded_image = np.zeros((padded_height, padded_width, channels), dtype=image.dtype)

    # Copy the image to the padded image
    padded_image[pad_height:pad_height + image_height, pad_width:pad_width + image_width, :] = image

    # Create the output image
    output_image = np.zeros((image_height, image_width, channels), dtype=image.dtype)

    # Apply the filter
    for row in range(image_height):
        for col in range(image_width):
            for channel in range(channels):
                output_image[row, col, channel] = np.sum(
                    np.multiply(kernel, padded_image[row:row + kernel_height, col:col + kernel_width, channel])) // (
                                                          kernel_height * kernel_width)

    return output_image


def vertical_gradient_filter(image):
    # Define a vertical gradient kernel
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

    # Perform convolution with the vertical gradient kernel
    gradient_image = cv2.filter2D(image, -1, kernel)

    return gradient_image


def horizontal_gradient_filter(image):
    # Define a horizontal gradient kernel
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)

    # Perform convolution with the horizontal gradient kernel
    gradient_image = cv2.filter2D(image, -1, kernel)

    return gradient_image


def gradient_filter_for_edge_detection(image):
    # Apply vertical gradient filter
    vertical_gradient = vertical_gradient_filter(image)

    # Apply horizontal gradient filter
    horizontal_gradient = horizontal_gradient_filter(image)

    # Calculate the magnitude of the gradient vectors
    edge_image = np.sqrt(np.square(vertical_gradient) + np.square(horizontal_gradient))

    # Normalize the image
    edge_image = edge_image / np.max(edge_image)

    # Convert the image to uint8
    edge_image = np.uint8(edge_image * 255)

    return edge_image


def minimum_filter(image, kernel_size):
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the padding required for the neighborhood
    pad = kernel_size // 2

    # Create an empty output image
    output_image = np.zeros_like(image)

    # Iterate through the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extract the neighborhood (kernel_size x kernel_size)
            neighborhood = image[i - pad:i + pad + 1, j - pad:j + pad + 1]

            # Find the minimum value within the neighborhood
            min_value = np.min(neighborhood)

            # Assign the minimum value to the corresponding pixel in the output image
            output_image[i, j] = min_value

    return output_image


def maximum_filter(image, kernel_size):
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the padding required for the neighborhood
    pad = kernel_size // 2

    # Create an empty output image
    output_image = np.zeros_like(image)

    # Iterate through the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extract the neighborhood (kernel_size x kernel_size)
            neighborhood = image[i - pad:i + pad + 1, j - pad:j + pad + 1]

            # Find the maximum value within the neighborhood
            max_value = np.max(neighborhood)

            # Assign the maximum value to the corresponding pixel in the output image
            output_image[i, j] = max_value

    return output_image


if __name__ == "__main__":
    print("Exercise 4")

    image = cv2.imread("Images_greyscale/Cervin.png")
    cv2.imshow("Cervin Original", image)
    print("Image shape: ", image.shape)

    # Part 1
    image_mean_filter_3 = convolutional_mean_filter(image, 3)
    cv2.imshow("Mean filter 3", image_mean_filter_3)
    cv2.imwrite("Output_Images/Cervin_Mean_Filter_3.png", image_mean_filter_3)
    image_mean_filter_3 = convolutional_mean_filter(image, 5)
    cv2.imshow("Mean filter 5", image_mean_filter_3)
    cv2.imwrite("Output_Images/Cervin_Mean_Filter_5.png", image_mean_filter_3)
    image_mean_filter_3 = convolutional_mean_filter(image, 9)
    cv2.imshow("Mean filter 9", image_mean_filter_3)
    cv2.imwrite("Output_Images/Cervin_Mean_Filter_9.png", image_mean_filter_3)

    # Part 2
    image = cv2.imread("Images_greyscale/Lena.png")
    cv2.imshow("Lena Original", image)
    print("Image shape: ", image.shape)
    kernel_logf = np.array([[0, 1, 1, 1, 0],
                            [1, 3, 0, 3, 1],
                            [1, 0, -24, 0, 1],
                            [1, 3, 0, 3, 1],
                            [0, 1, 1, 1, 0]])
    image_laplacian_of_gaussian_filter = laplacian_of_gaussian_filter(image, kernel_logf)
    cv2.imshow("Laplacian of Gaussian filter", image_laplacian_of_gaussian_filter)
    cv2.imwrite("Output_Images/Lena_Laplacian_of_Gaussian_Filter.png", image_laplacian_of_gaussian_filter)
    print("Image shape: ", image_laplacian_of_gaussian_filter.shape)
    image_gradient_filter = gradient_filter_for_edge_detection(image)
    cv2.imshow("Gradient filter", image_gradient_filter)
    cv2.imwrite("Output_Images/Lena_Gradient_Filter.png", image_gradient_filter)

    # Part 3
    image = cv2.imread("Images_greyscale/Stop.png")
    cv2.imshow("Beetle Original", image)
    print("Image shape: ", image.shape)
    image_minimum_filter = minimum_filter(image, 3)
    cv2.imshow("Minimum filter", image_minimum_filter)
    cv2.imwrite("Output_Images/Stop_Minimum_Filter.png", image_minimum_filter)
    image_maximum_filter = maximum_filter(image, 3)
    cv2.imshow("Maximum filter", image_maximum_filter)
    cv2.imwrite("Output_Images/Stop_Maximum_Filter.png", image_maximum_filter)
    image_max_min = image_maximum_filter - image_minimum_filter
    cv2.imshow("Max - min", image_max_min)
    cv2.imwrite("Output_Images/Stop_Max_Min.png", image_max_min)

    cv2.waitKey(0)
