import cv2
import numpy as np
from noise import add_salt_pepper_noise, add_gaussian_noise

# def adjust_contrast(image, factor):
#     return np.clip((image - 128) * factor + 128, 0, 255).astype(np.uint8)

def adjust_contrast(image, factor):
    """ Adjusts the contrast of an image by scaling the intensity of the pixels.
        factor > 1 increases contrast, factor < 1 decreases it.
    """
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


# def create_contrasted_noised_images(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     high_contrast = adjust_contrast(image, 2.0)  # Increase contrast
#     low_contrast = adjust_contrast(image, 0.5)  # Decrease contrast
#     high_sp_noise = add_salt_pepper_noise(high_contrast)
#     low_sp_noise = add_salt_pepper_noise(low_contrast)
#     high_gauss_noise = add_gaussian_noise(high_contrast)
#     low_gauss_noise = add_gaussian_noise(low_contrast)
#     return high_sp_noise, low_sp_noise, high_gauss_noise, low_gauss_noise


def create_contrasted_noised_images(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    high_contrast = adjust_contrast(image, 1.5)  # Enhanced contrast
    low_contrast = adjust_contrast(image, 0.5)  # Reduced contrast

    high_sp_noise = add_salt_pepper_noise(high_contrast)
    low_sp_noise = add_salt_pepper_noise(low_contrast)
    high_gauss_noise = add_gaussian_noise(high_contrast)
    low_gauss_noise = add_gaussian_noise(low_contrast)

    cv2.imwrite('noised/high_contrast/high_sp.png', high_sp_noise)
    cv2.imwrite('noised/low_contrast/low_sp.png', low_sp_noise)
    cv2.imwrite('noised/high_contrast/high_gauss.png', high_gauss_noise)
    cv2.imwrite('noised/low_contrast/low_gauss.png', low_gauss_noise)

    return high_sp_noise, low_sp_noise, high_gauss_noise, low_gauss_noise