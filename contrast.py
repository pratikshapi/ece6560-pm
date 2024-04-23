import cv2
import numpy as np
from noise import add_salt_pepper_noise, add_gaussian_noise


def adjust_contrast(image, factor):
    """ Adjusts the contrast of an image by scaling the intensity of the pixels.
        factor > 1 increases contrast, factor < 1 decreases it.
    """
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

def create_contrasted_noised_images(image):
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    high_contrast = adjust_contrast(image, 1.5)  # Enhanced contrast
    low_contrast = adjust_contrast(image, 0.5)  # Reduced contrast

    return high_contrast, low_contrast