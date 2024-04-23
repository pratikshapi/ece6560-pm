import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import logging

from noise import add_salt_pepper_noise, add_gaussian_noise
from denoise import wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2
from metrics import psnr, mse
from util import ensure_dir
from contrast import create_contrasted_noised_images

# Set up logging
logging.basicConfig(filename='denoise.log', level=logging.INFO)

def optimize_pmf_parameters(image, K_values, step_counts, func):
    best_psnr, best_mse, best_K, best_steps = -np.inf, np.inf, None, None
    original_image = cv2.imread('images/original.jpg', cv2.IMREAD_GRAYSCALE)  # Path to the original image for PSNR comparison
    for K in K_values:
        for steps in step_counts:
            denoised_image = func(image, steps, K)
            current_psnr = psnr(original_image, denoised_image)
            current_mse = mse(original_image, denoised_image)
            logging.info(f'K: {K}, Steps: {steps}, PSNR: {current_psnr}, MSE: {current_mse}')
            if current_psnr > best_psnr:
                best_psnr, best_mse = current_psnr, current_mse
                best_K, best_steps = K, steps
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

def process_and_save_images(original_image, noise_funcs, noise_labels, denoise_funcs, denoise_labels, directory=""):
    noisy_images = [func(original_image.copy()) for func in noise_funcs]

    for noisy, noise_label in zip(noisy_images, noise_labels):
        noisy_filename = f'noised/{directory}{noise_label}.png'
        cv2.imwrite(noisy_filename, noisy)

        for denoise_func, denoise_label in zip(denoise_funcs, denoise_labels):
            if 'perona_malik' in denoise_label:
                best_K, best_steps, _, _ = optimize_pmf_parameters(noisy, K_values, step_counts, denoise_func)
                denoised_image = denoise_func(noisy, best_steps, best_K)
            else:
                denoised_image = denoise_func(noisy)

            denoised_filename = f'denoised/{directory}{noise_label}_with_{denoise_label}.png'
            cv2.imwrite(denoised_filename, denoised_image)

def validate_image(image):
    if image is None or len(image.shape) != 2:
        raise ValueError("Invalid image. Ensure it is grayscale and exists.")

def main():
    image_path = 'images/dog.jpg'
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)

    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']
    denoise_funcs = [wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2]
    denoise_labels = ['wavelet_denoising', 'gaussian_denoising', 'perona_malik_f1_denoising', 'perona_malik_f2_denoising']

    process_and_save_images(original_image, noise_funcs, noise_labels, denoise_funcs, denoise_labels)

def main_contrast():
    image_path = 'images/dog.jpg'
    high_sp, low_sp, high_gauss, low_gauss = create_contrasted_noised_images(image_path)

    noise_images = [high_sp, low_sp, high_gauss, low_gauss]
    noise_labels = ['high_sp', 'low_sp', 'high_gauss', 'low_gauss']
    denoise_funcs = [wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2]
    denoise_labels = ['wavelet_denoising', 'gaussian_denoising', 'perona_malik_f1_denoising', 'perona_malik_f2_denoising']

    # High and low contrast images directory setup
    directories = ['high_contrast/', 'low_contrast/', 'high_contrast/', 'low_contrast/']
    for img, label, directory in zip(noise_images, noise_labels, directories):
        process_and_save_images(img, [lambda x: x], [label], denoise_funcs, denoise_labels, directory)


if __name__ == '__main__':

    image_path = 'images/dog.jpg'
    K_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 10]
    step_counts = [5, 10, 50, 100, 150, 200]
    ensure_dirs()
    main()
    main_contrast()