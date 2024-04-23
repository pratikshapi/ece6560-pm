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


# def plot_results(original, noisy, denoised, noise_labels, denoise_labels):
#     plt.figure(figsize=(15, 10))
#     num_noisy = len(noisy)
#     cols = num_noisy + 1  # For noisy images and the original image

#     # First row: noisy images first, then the original image
#     for i, img in enumerate(noisy, start=1):
#         plt.subplot(5, cols, i)
#         plt.imshow(img, cmap='gray')
#         plt.title(noise_labels[i-1])
#         plt.axis('off')

#     # Place the original image last in the first row
#     plt.subplot(5, cols, cols)
#     plt.imshow(original, cmap='gray')
#     plt.title("Original")
#     plt.axis('off')

#     # Subsequent rows: denoised images for each method
#     row_offset = cols
#     for d_index, denoised_group in enumerate(denoised):
#         for i, img in enumerate(denoised_group, start=row_offset + 1):
#             plt.subplot(5, cols, i)
#             plt.imshow(img, cmap='gray')
#             plt.title(f"{denoise_labels[d_index]}\n{noise_labels[(i - row_offset - 1) % num_noisy]}")
#             plt.axis('off')
#         row_offset += cols

#     plt.tight_layout()
#     plt.show()

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

# def process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels, display=True):
#     original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     noisy_images = []
    
#     # Create and save noisy images
#     for func, label in zip(noise_funcs, noise_labels):
#         noisy = func(original_image.copy())
#         noisy_images.append(noisy)
#         cv2.imwrite(f'noised/{label.replace(" ", "_").lower()}.png', noisy)
    
#     # Create and save denoised images
#     denoised_images = []
#     for noisy, n_label in zip(noisy_images, noise_labels):
#         for func, d_label in zip(denoise_funcs, denoise_labels):
#             denoised = func(noisy.copy())
#             denoised_images.append(denoised)
#             filename = f'{n_label.replace(" ", "_").lower()}_with_{d_label.replace(" ", "_").lower()}.png'
#             cv2.imwrite(f'denoised/{filename}', denoised)

    
#     if display:
#         plot_results(original_image, noisy_images, denoised_images, noise_labels, denoise_labels)

#     return noisy_images, denoised_images


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


# ensure_dir('noised')
# ensure_dir('denoised')

# def main():
#     image_path = 'images/dog.jpg'
#     noise_funcs = [
#         lambda img: add_salt_pepper_noise(img, prob=0.05),  # Replacing speckle noise
#         lambda img: add_gaussian_noise(img, var=0.05)  # Assuming Gaussian noise remains the same
#     ]
#     noise_labels = ['Salt & Pepper Noise', 'Gaussian Noise']  # Update labels accordingly

    
#     denoise_funcs = [
#         lambda img: wavelet_denoising(img, level=3),
#         lambda img: gaussian_denoise(img, sigma=1.0),
#         lambda img: anisodiff_f1(img, steps=50, K=0.1),
#         lambda img: anisodiff_f2(img, steps=50, K=0.1)
#     ]
#     denoise_labels = [
#         'Wavelet Denoising',
#         'Gaussian Denoising',
#         'Perona-Malik F1 Denoising',
#         'Perona-Malik F2 Denoising'
#     ]

#     process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels)

# if __name__ == '__main__':
#     main()