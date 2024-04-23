import os
import cv2

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

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
