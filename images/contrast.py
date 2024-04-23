import cv2
import numpy as np
# from noise import add_salt_pepper_noise, add_gaussian_noise


def adjust_contrast(image, factor):
    """ Adjusts the contrast of an image by scaling the intensity of the pixels.
        factor > 1 increases contrast, factor < 1 decreases it.
    """
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

def create_contrasted_noised_images(image_path='./original.jpg'):

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(original_image)

    # Generate high and low contrast versions of the image
    high_contrast_image = adjust_contrast(original_image, 1.5)
    low_contrast_image = adjust_contrast(original_image, 0.5)

    # Save the contrast-adjusted images
    high_contrast_path = image_path.replace('.jpg', '_high_contrast.jpg')
    low_contrast_path = image_path.replace('.jpg', '_low_contrast.jpg')
    
    cv2.imwrite(high_contrast_path, high_contrast_image)
    cv2.imwrite(low_contrast_path, low_contrast_image)

    return high_contrast_path, low_contrast_path


create_contrasted_noised_images()
