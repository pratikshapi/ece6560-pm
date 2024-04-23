import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import os

def add_speckle_noise(image, mean=0, var=0.1):
    row, col = image.shape
    gauss = np.random.randn(row, col)
    noisy = image + image * gauss * var
    return noisy.astype('uint8')

def add_gaussian_noise(image, mean=0, var=0.1):
    row, col = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col)) * 50
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype('uint8')

def wavelet_denoising(image, wavelet='db4', level=5):
    coeffs = pywt.wavedec2(image, wavelet, mode='per', level=level)
    sigma = (1/0.6745) * madev(coeffs[-1][0])
    uthresh = sigma * np.sqrt(2 * np.log(image.size))
    new_coeffs = [coeffs[0]] + [tuple(pywt.threshold(detail, value=uthresh, mode='hard') for detail in level) for level in coeffs[1:]]
    denoised_image = pywt.waverec2(new_coeffs, wavelet, mode='per')
    return denoised_image

def gaussian_denoise(image, kernel_size=5, sigma=1.5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def madev(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def psnr(target, ref):
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mse(target, ref):
    return np.mean((target - ref) ** 2)

def plot_results(images, titles):
    plt.figure(figsize=(10, 5))
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels, display=True):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    denoised_images = [func(noisy) for noisy in noisy_images for func in denoise_funcs]
    
    if display:
        titles = ["Original"] + [label for label in noise_labels] + [f"{n_label} + {d_label}" for n_label in noise_labels for d_label in denoise_labels]
        plot_results([original_image] + noisy_images + denoised_images, titles)
    return noisy_images, denoised_images

def main():
    image_path = 'images/dog.jpg'
    noise_funcs = [
        lambda img: add_speckle_noise(img, var=0.05),
        lambda img: add_gaussian_noise(img, var=0.05)
    ]
    noise_labels = [
        "Speckle Noise",
        "Gaussian Noise"
    ]
    denoise_funcs = [
        lambda img: wavelet_denoising(img, level=3),
        lambda img: gaussian_denoise(img, sigma=1.0)
    ]
    denoise_labels = [
        "Wavelet Denoise",
        "Gaussian Denoise"
    ]

    noisy_images, denoised_images = process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels)

# def process_image(image_path, noise_funcs, denoise_funcs, display=True):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    denoised_images = [func(noisy) for noisy in noisy_images for func in denoise_funcs]

    if display:
        titles = ["Original"] + ["Noisy" for _ in range(len(noise_funcs))] + ["Denoised" for _ in range(len(denoise_funcs))]
        plot_results([original_image] + noisy_images + denoised_images, titles)
    return noisy_images, denoised_images

# def main():
    image_path = 'images/dog.jpg'
    noise_funcs = [
        lambda img: add_speckle_noise(img, var=0.05),
        lambda img: add_gaussian_noise(img, var=0.05)
    ]
    denoise_funcs = [
        lambda img: wavelet_denoising(img, level=3),
        lambda img: gaussian_denoise(img, sigma=1.0)
    ]

    noisy_images, denoised_images = process_image(image_path, noise_funcs, denoise_funcs)

if __name__ == '__main__':
    main()