import cv2
import numpy as np


def separate_rgb_channels(img):
    b, g, r = cv2.split(img)
    cv2.imshow("Blue", b)
    cv2.imwrite("Output_Images/Lena_Blue.png", b)
    cv2.imshow("Green", g)
    cv2.imwrite("Output_Images/Lena_Green.png", g)
    cv2.imshow("Red", r)
    cv2.imwrite("Output_Images/Lena_Red.png", r)
    return [r, g, b]


def rgb_to_hsl(img):
    output_image = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            [b, g, r] = img[i, j]
            r = r / 255
            g = g / 255
            b = b / 255
            c_max = max(r, max(g, b))
            c_min = min(r, min(g, b))
            delta = c_max - c_min

            # Hue
            if delta == 0:
                h = 0
            elif c_max == r:
                h = 60 * (((g - b) / delta) % 6)
            elif c_max == g:
                h = 60 * (((b - r) / delta) + 2)
            elif c_max == b:
                h = 60 * (((r - g) / delta) + 4)

            # Lightness
            l = (c_max + c_min) / 2

            # Saturation
            if delta == 0:
                s = 0
            else:
                s = delta / (1 - abs(2 * l - 1))

            output_image[i, j] = [h, s, l]

    return output_image


def reconstruct_rgb_from_hsl(img):
    output_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            [h, s, l] = img[i, j]

            c = (1 - abs(2 * l - 1)) * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = l - c / 2

            if 0 <= h < 60:
                r, g, b = c, x, 0
            elif 60 <= h < 120:
                r, g, b = x, c, 0
            elif 120 <= h < 180:
                r, g, b = 0, c, x
            elif 180 <= h < 240:
                r, g, b = 0, x, c
            elif 240 <= h < 300:
                r, g, b = x, 0, c
            elif 300 <= h < 360:
                r, g, b = c, 0, x

            r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255

            output_image[i, j] = [r, g, b]

    return output_image


def greyscale_histogram_equalization(img):
    # Compute the histogram of the input grayscale image
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Compute the cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()

    # Normalize the CDF to have values in the range [0, 255]
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Map the intensity values in the original image to their equalized values
    equalized_image = cdf_normalized[img]

    # Convert the resulting image to 8-bit unsigned integer format
    equalized_image = equalized_image.astype(np.uint8)

    return equalized_image


if __name__ == "__main__":
    print("Exercise 3")

    # Part 1
    image = cv2.imread("Images/Lena.png")
    cv2.imshow("Original", image)
    cv2.imwrite("Output_Images/Lena_Original.png", image)
    hsl_image = rgb_to_hsl(image)
    cv2.imshow("HSL", hsl_image)
    cv2.imwrite("Output_Images/Lena_HSL.png", hsl_image)
    rgb_image = reconstruct_rgb_from_hsl(hsl_image)
    cv2.imshow("RGB", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("Output_Images/Lena_RGB.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    # Part 2
    greyscale_image = cv2.imread("Images_greyscale/Lena.png", 0)
    cv2.imshow("Greyscale", greyscale_image)
    cv2.imwrite("Output_Images/Lena_Greyscale.png", greyscale_image)
    equalized_image = greyscale_histogram_equalization(greyscale_image)
    cv2.imshow("Equalized", equalized_image)
    cv2.imwrite("Output_Images/Lena_Greyscale_Equalized.png", equalized_image)

    # Part 3
    [ch_r, ch_g, ch_b] = separate_rgb_channels(image)
    eq_ch_r = greyscale_histogram_equalization(ch_r)
    eq_ch_g = greyscale_histogram_equalization(ch_g)
    eq_ch_b = greyscale_histogram_equalization(ch_b)
    cv2.imshow("Reconstructed RGB", cv2.merge([eq_ch_b, eq_ch_g, eq_ch_r]))
    cv2.imwrite("Output_Images/Lena_RGB_Equalized.png", cv2.merge([eq_ch_b, eq_ch_g, eq_ch_r]))

    [ch_h, ch_s, ch_l] = cv2.split(hsl_image)
    eq_ch_l = cv2.equalizeHist((ch_l*255).astype(np.uint8))
    merged_hsl_image = cv2.merge([ch_h, ch_s, eq_ch_l.astype(np.float32) / 255])
    cv2.imshow("Reconstructed HSL", cv2.cvtColor(reconstruct_rgb_from_hsl(merged_hsl_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("Output_Images/Lena_HSL_Equalized.png", cv2.cvtColor(reconstruct_rgb_from_hsl(merged_hsl_image), cv2.COLOR_RGB2BGR))

    cv2.waitKey(0)
