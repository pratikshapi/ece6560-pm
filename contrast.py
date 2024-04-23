import cv2
import numpy as np
from noise import add_salt_pepper_noise, add_gaussian_noise

def adjust_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def create_contrasted_noised_images(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    high_contrast = adjust_contrast(image, 2.0)  # Increase contrast
    low_contrast = adjust_contrast(image, 0.5)  # Decrease contrast
    high_sp_noise = add_salt_pepper_noise(high_contrast)
    low_sp_noise = add_salt_pepper_noise(low_contrast)
    high_gauss_noise = add_gaussian_noise(high_contrast)
    low_gauss_noise = add_gaussian_noise(low_contrast)
    return high_sp_noise, low_sp_noise, high_gauss_noise, low_gauss_noise
