import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import logging

from noise import add_salt_pepper_noise, add_gaussian_noise
from denoise import wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2
from metrics import psnr, mse
from util import ensure_dir

# Set up logging
logging.basicConfig(filename='denoise.log', level=logging.INFO)

def optimize_pmf_parameters(noisy_image_path, original_image_path, func, K_values, step_counts):
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    best_psnr, best_mse, best_K, best_steps = -np.inf, np.inf, None, None
    
    for K in K_values:
        for steps in step_counts:
            denoised_image = func(noisy_image, steps, K)
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

def process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels, display=True):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError("Image not found or the path is incorrect")
    if len(original_image.shape) != 2:
        raise ValueError("Image loaded is not in grayscale")

    noisy_images = []
    for func, label in zip(noise_funcs, noise_labels):
        noisy = func(original_image.copy())
        if len(noisy.shape) != 2:
            raise ValueError(f"Noisy image processing failed for {label}, resulting in non-2D shape")
        noisy_images.append(noisy)
        filename_noisy = f'noised/{label.replace(" ", "_").lower()}.png'
        cv2.imwrite(filename_noisy, noisy)
    
    denoised_images = []
    for noisy, n_label in zip(noisy_images, noise_labels):
        for func, d_label in zip(denoise_funcs, denoise_labels):
            denoised = func(noisy.copy())
            if len(denoised.shape) != 2:
                raise ValueError(f"Denoising failed for {d_label} applied on {n_label}, resulting in non-2D shape")
            denoised_images.append(denoised)
            filename_denoised = f'denoised/{n_label.replace(" ", "_").lower()}_with_{d_label.replace(" ", "_").lower()}.png'
            cv2.imwrite(filename_denoised, denoised)

    if display:
        plot_results(original_image, noisy_images, denoised_images, noise_labels, denoise_labels)

def main():
    ensure_dir('noised')
    ensure_dir('denoised')
    image_path = 'images/dog.jpg'
    
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt & pepper noise', 'gaussian noise'] 
    denoise_funcs = [wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2]
    denoise_labels = ['wavelet denoising', 'gaussian denoising', 'perona-malik f1 denoising', 'perona-malik f2 denoising']
    
    process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels)


if __name__ == '__main__':
    main()