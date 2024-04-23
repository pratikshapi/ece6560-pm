import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import logging

from noise import add_salt_pepper_noise, add_gaussian_noise
from denoise import wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2
from metrics import psnr, nmse
from util import ensure_dir, create_directory_structure
from contrast import create_contrasted_noised_images

# Set up logging
logging.basicConfig(filename='denoise.log', level=logging.INFO)

def setup_logger():
    logger = logging.getLogger('PSNR_Results')
    handler = logging.FileHandler('results.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

results_logger = setup_logger()

def log_psnr_results(original_image, denoised_images, labels, logger):
    for denoised_image, label in zip(denoised_images, labels):
        psnr_value = psnr(original_image, denoised_image)
        logger.info(f"{label}: PSNR = {psnr_value}")


def optimize_pmf_parameters(image, K_values, step_counts, func):
    best_psnr, best_mse, best_K, best_steps = -np.inf, np.inf, None, None
    original_image = cv2.imread('images/original.jpg', cv2.IMREAD_GRAYSCALE)  # Path to the original image for PSNR comparison
    for K in K_values:
        for steps in step_counts:
            denoised_image = func(image, steps, K)
            current_psnr = psnr(original_image, denoised_image)
            current_mse = nmse(original_image, denoised_image)
            if current_psnr > best_psnr:
                best_psnr, best_mse = current_psnr, current_mse
                best_K, best_steps = K, steps
    logging.info(f'best_K: {best_K}, best_steps: {best_steps}, best_psnr: {best_psnr}, best_mse: {best_mse}')
    return best_K, best_steps, best_psnr, best_mse

def plot_results(original, noisy, denoised, noise_labels, denoise_labels):
    plt.figure(figsize=(15, 10))
    images = [original] + noisy + denoised
    titles = ['Original'] + [label for label in noise_labels] + [f"{n_label} with {d_label}" for n_label in noise_labels for d_label in denoise_labels]

    for img, title in zip(images, titles):
        if len(img.shape) != 2:
            raise ValueError(f"Attempting to plot image with invalid shape: {img.shape}")
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_image(original_image, noise_funcs, denoise_funcs):
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    denoised_images = []
    for noisy in noisy_images:
        denoised_images.extend([func(noisy.copy()) for func in denoise_funcs])
    return noisy_images, denoised_images

def ensure_dirs():
    dirs = ['noised', 'denoised', 'noised/high_contrast', 'noised/low_contrast', 'denoised/high_contrast', 'denoised/low_contrast']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def save_image(image, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, filename), image)

def process_and_save_denoised_images(image_path, noise_type, contrast_type, denoise_func, param, param_value):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if 'varyingK' in denoise_func.__name__:
        denoised_image = denoise_func(image, K=param_value)
    elif 'varyingDelt' in denoise_func.__name__:
        denoised_image = denoise_func(image, del_t=param_value)

    directory = f'denoised/{contrast_type}/{denoise_func.__name__}/{noise_type}'
    filename = f'{param}_{param_value}.png'
    save_image(denoised_image, directory, filename)

def process_with_varying_k(noisy_image, noise_type):
    for K in K_values:
        denoised_f1 = anisodiff_f1(noisy_image.copy(), K=K, steps=10, del_t=0.1)
        denoised_f2 = anisodiff_f2(noisy_image.copy(), K=K, steps=50, del_t=0.1)
        save_image(denoised_f1, f'denoised/varyingK/{noise_type}', f'f1_K_{K}.png')
        save_image(denoised_f2, f'denoised/varyingK/{noise_type}', f'f2_K_{K}.png')

def process_with_varying_delt(noisy_image, noise_type):
    for del_t in del_t_values:
        denoised_f1 = anisodiff_f1(noisy_image.copy(), K=0.1, steps=10, del_t=del_t)
        denoised_f2 = anisodiff_f2(noisy_image.copy(), K=0.1, steps=50, del_t=del_t)
        save_image(denoised_f1, f'denoised/varyingDelt/{noise_type}', f'f1_del_t_{del_t}.png')
        save_image(denoised_f2, f'denoised/varyingDelt/{noise_type}', f'f2_del_t_{del_t}.png')

def process_and_save_images(noisy_images, noise_labels, denoise_funcs, denoise_labels, base_directory="", K_values=None, step_counts=None):
    for noisy, noise_label in zip(noisy_images, noise_labels):
        # Update the directory path based on the noise label
        directory = f"{base_directory}{noise_label}/"
        
        for denoise_func, denoise_label in zip(denoise_funcs, denoise_labels):
            if 'perona_malik' in denoise_label:
                # Perform grid search for optimal parameters
                best_psnr, best_image, best_K, best_steps = -float('inf'), None, None, None
                for K in K_values:
                    for steps in step_counts:
                        denoised_image = denoise_func(noisy.copy(), steps=steps, K=K)
                        psnr_value = psnr(original_image, denoised_image)  # Ensure 'original_image' is accessible
                        if psnr_value > best_psnr:
                            best_psnr = psnr_value
                            best_image = denoised_image
                            best_K = K
                            best_steps = steps
                # Save best denoised image from grid search
                denoised_filename = f'{directory}{noise_label}_with_{denoise_label}_K{best_K}_steps{best_steps}.png'
                cv2.imwrite(denoised_filename, best_image)
            else:
                # Apply denoising method directly without parameter tuning
                denoised_image = denoise_func(noisy.copy())
                denoised_filename = f'{directory}{noise_label}_with_{denoise_label}.png'
                cv2.imwrite(denoised_filename, denoised_image)
        
        log_psnr_results(original_image, denoised_images, labels, logger)


# def process_and_save_images(noisy_images, noise_labels, denoise_funcs, denoise_labels, directory="", K_values=None, step_counts=None):
#     for noisy, noise_label in zip(noisy_images, noise_labels):
#         for denoise_func, denoise_label in zip(denoise_funcs, denoise_labels):
#             if 'perona_malik' in denoise_label:
#                 # Perform grid search for optimal parameters
#                 best_psnr, best_image, best_K, best_steps = -float('inf'), None, None, None
#                 for K in K_values:
#                     for steps in step_counts:
#                         denoised_image = denoise_func(noisy.copy(), steps=steps, K=K)
#                         psnr_value = psnr(original_image, denoised_image)
#                         if psnr_value > best_psnr:
#                             best_psnr = psnr_value
#                             best_image = denoised_image
#                             best_K = K
#                             best_steps = steps
#                 # Save best denoised image from grid search
#                 denoised_filename = f'denoised/{directory}{noise_label}_with_{denoise_label}_K{best_K}_steps{best_steps}.png'
#                 cv2.imwrite(denoised_filename, best_image)
#             else:
#                 # Apply denoising method directly without parameter tuning
#                 denoised_image = denoise_func(noisy.copy())
#                 denoised_filename = f'denoised/{directory}{noise_label}_with_{denoise_label}.png'
#                 cv2.imwrite(denoised_filename, denoised_image)

# def process_and_save_images(original_image, noise_funcs, noise_labels, denoise_funcs, denoise_labels, directory=""):
#     noisy_images = [func(original_image.copy()) for func in noise_funcs]

#     for noisy, noise_label in zip(noisy_images, noise_labels):
#         noisy_filename = f'noised/{directory}{noise_label}.png'
#         cv2.imwrite(noisy_filename, noisy)

#         for denoise_func, denoise_label in zip(denoise_funcs, denoise_labels):
#             if 'perona_malik' in denoise_label:
#                 best_K, best_steps, _, _ = optimize_pmf_parameters(noisy, K_values, step_counts, denoise_func)
#                 denoised_image = denoise_func(noisy, best_steps, best_K)
#             else:
#                 denoised_image = denoise_func(noisy)

#             denoised_filename = f'denoised/{directory}{noise_label}_with_{denoise_label}.png'
#             cv2.imwrite(denoised_filename, denoised_image)

# def process_and_save_images(noisy_images, noise_labels, denoise_funcs, denoise_labels, directory=""):
#     for noisy, noise_label in zip(noisy_images, noise_labels):
#         for denoise_func, denoise_label in zip(denoise_funcs, denoise_labels):
#             denoised_image = denoise_func(noisy.copy())
#             denoised_filename = f'denoised/{directory}{noise_label}_with_{denoise_label}.png'
#             cv2.imwrite(denoised_filename, denoised_image)


def validate_image(image):
    if image is None or len(image.shape) != 2:
        raise ValueError("Invalid image. Ensure it is grayscale and exists.")

def generate_noised_images_if_absent(original_image_path, noise_funcs, noise_labels, contrast_levels):
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)
    
    # Check and create noised images for each contrast level
    for contrast in contrast_levels:
        for func, label in zip(noise_funcs, noise_labels):
            noised_image_path = f'noised/{contrast}/{label}.png'
            if not os.path.exists(noised_image_path):
                if contrast == 'normal':
                    noised_image = func(original_image.copy())
                else:
                    # Assuming create_contrasted_noised_images is a function that takes an image and a contrast label
                    # and returns a noised image with the specified contrast level
                    noised_image = create_contrasted_noised_images(original_image, contrast, func)
                save_image(noised_image, f'noised/{contrast}', f'{label}.png')

def generate_noisy_images(original_image, noise_funcs):
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    return noisy_images

def main():
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)
    base_directory = "denoised/"
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']
    generate_noised_images_if_absent(image_path, noise_funcs, noise_labels, ['normal'])
    denoise_funcs = [wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2]
    denoise_labels = ['wavelet_denoising', 'gaussian_denoising', 'perona_malik_f1_denoising', 'perona_malik_f2_denoising']

    noisy_images = generate_noisy_images(original_image, noise_funcs)
    process_and_save_images(noisy_images, noise_labels, denoise_funcs, denoise_labels, base_directory, K_values, step_counts)

def main_contrast():
    high_sp, low_sp, high_gauss, low_gauss = create_contrasted_noised_images(image_path)
    generate_noised_images_if_absent(image_path, noise_funcs, noise_labels, contrast_levels)

    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_images = [high_sp, low_sp, high_gauss, low_gauss]
    noise_labels = ['high_sp', 'low_sp', 'high_gauss', 'low_gauss']
    denoise_funcs = [wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2]
    denoise_labels = ['wavelet_denoising', 'gaussian_denoising', 'perona_malik_f1_denoising', 'perona_malik_f2_denoising']

    # High and low contrast images directory setup
    directories = ['high_contrast/', 'low_contrast/', 'high_contrast/', 'low_contrast/']
    for img, label, directory in zip(noise_images, noise_labels, directories):
        process_and_save_images(img, [lambda x: x], [label], denoise_funcs, denoise_labels, directory)

def main_varying_k():
    generate_noised_images_if_absent(image_path, noise_funcs, noise_labels, ['normal'])
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']



    gaussian_image = cv2.imread('noised/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)
    salt_pepper_image = cv2.imread('noised/salt_pepper_noise.png', cv2.IMREAD_GRAYSCALE)

    process_with_varying_k(gaussian_image, 'gaussian')
    process_with_varying_k(salt_pepper_image, 'salt_pepper')

def main_varing_delt():
    generate_noised_images_if_absent(image_path, noise_funcs, noise_labels, ['normal'])
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']

    gaussian_image = cv2.imread('noised/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)
    salt_pepper_image = cv2.imread('noised/salt_pepper_noise.png', cv2.IMREAD_GRAYSCALE)

    process_with_varying_delt(gaussian_image, 'gaussian')
    process_with_varying_delt(salt_pepper_image, 'salt_pepper')

def generate_all_noised_images(original_image_path, noise_funcs, noise_labels, contrast_levels):
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)

    # Handle the normal contrast level first
    for func, label in zip(noise_funcs, noise_labels):
        noised_image_path = f'noised/normal/{label}.png'
        if not os.path.exists(noised_image_path):
            noised_image = func(original_image.copy())
            save_image(noised_image, 'noised/normal', f'{label}.png')

    # Now handle high and low contrast levels
    high_contrast, low_contrast = create_contrasted_noised_images(original_image)
    contrasts = {'high_contrast': high_contrast, 'low_contrast': low_contrast}
    for contrast_name, contrasted_image in contrasts.items():
        for func, label in zip(noise_funcs, noise_labels):
            noised_image_path = f'noised/{contrast_name}/{label}.png'
            if not os.path.exists(noised_image_path):
                noised_image = func(contrasted_image.copy())
                save_image(noised_image, f'noised/{contrast_name}', f'{label}.png')


if __name__ == '__main__':

    image_path = 'images/dog.jpg'
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    K_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 10]
    del_t_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    step_counts = [5, 10, 50, 100, 150, 200]
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']
    contrast_levels = ['high_contrast', 'low_contrast', 'normal']

    # create_directory_structure()
    # generate_all_noised_images(image_path, noise_funcs, noise_labels, contrast_levels)

    main()
    # main_contrast()
    # main_varying_k()
    # main_varing_delt()