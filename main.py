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

def F1(dt, K):
    return np.exp(-1 * (np.power(dt, 2)) / (np.power(K, 2)))

def F2(dt, K):
    func = 1 / (1 + ((dt/K)**2))
    return func

def anisodiff_f1(img, steps, K, del_t=0.25):
    upgrade_img = np.zeros(img.shape, dtype=img.dtype)
    for t in range(steps):
        dn = img[:-2, 1:-1] - img[1:-1, 1:-1]
        ds = img[2:, 1:-1] - img[1:-1, 1:-1]
        de = img[1:-1, 2:] - img[1:-1, 1:-1]
        dw = img[1:-1, :-2] - img[1:-1, 1:-1]
        upgrade_img[1:-1, 1:-1] = img[1:-1, 1:-1] + del_t * (
            F1(dn, K) * dn + F1(ds, K) * ds + F1(de, K) * de + F1(dw, K) * dw)
        img = upgrade_img
    return img

def anisodiff_f2(img, steps, K, del_t=0.25):
    upgrade_img = np.zeros(img.shape, dtype=img.dtype)
    for t in range(steps):
        dn = img[:-2, 1:-1] - img[1:-1, 1:-1]
        ds = img[2:, 1:-1] - img[1:-1, 1:-1]
        de = img[1:-1, 2:] - img[1:-1, 1:-1]
        dw = img[1:-1, :-2] - img[1:-1, 1:-1]
        upgrade_img[1:-1, 1:-1] = img[1:-1, 1:-1] + del_t * (
            F2(dn, K) * dn + F2(ds, K) * ds + F2(de, K) * de + F2(dw, K) * dw)
        img = upgrade_img
    return img

def psnr(target, ref):
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mse(target, ref):
    return np.mean((target - ref) ** 2)


def plot_results(original, noisy, denoised, noise_labels, denoise_labels):
    plt.figure(figsize=(15, 10))
    num_noisy = len(noisy)
    cols = max(2, num_noisy + 1)  # At least two columns for original and one type of noisy image

    # First row: original and noisy images
    plt.subplot(5, cols, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i, img in enumerate(noisy, start=2):
        plt.subplot(5, cols, i)
        plt.imshow(img, cmap='gray')
        plt.title(noise_labels[i-2])
        plt.axis('off')

    # Subsequent rows: denoised images for each method
    row_offset = cols
    for d_index, (denoised_group, label) in enumerate(zip(denoised, denoise_labels), start=1):
        for i, img in enumerate(denoised_group, start=row_offset + 1):
            plt.subplot(5, cols, i)
            plt.imshow(img, cmap='gray')
            plt.title(f"{label}\n{noise_labels[(i - row_offset - 1) % num_noisy]}")
            plt.axis('off')
        row_offset += cols

    plt.tight_layout()
    plt.show()


# def plot_results(original, noisy, denoised_wavelet, denoised_gaussian, denoised_pm_f1, denoised_pm_f2, noise_labels):
    plt.figure(figsize=(15, 10))
    num_noisy = len(noisy)
    cols = max(2, num_noisy + 1)  # At least two columns for original and one type of noisy image
    
    # Plot original and all noisy images in the first row
    plt.subplot(5, cols, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    for i, img in enumerate(noisy, start=2):
        plt.subplot(5, cols, i)
        plt.imshow(img, cmap='gray')
        plt.title(noise_labels[i-2])
        plt.axis('off')
    
    # Plot denoised images row by row
    def plot_denoised(row_start, denoised_images):
        for i, img in enumerate(denoised_images, start=row_start):
            plt.subplot(5, cols, i)
            plt.imshow(img, cmap='gray')
            plt.title(f"{noise_labels[(i-row_start)%num_noisy]}")
            plt.axis('off')
    
    plot_denoised(cols + 1, denoised_wavelet)
    plot_denoised(2 * cols + 1, denoised_gaussian)
    plot_denoised(3 * cols + 1, denoised_pm_f1)
    plot_denoised(4 * cols + 1, denoised_pm_f2)

    plt.tight_layout()
    plt.show()


# def plot_results(images, titles):
#     plt.figure(figsize=(10, 5))
#     for i, (img, title) in enumerate(zip(images, titles), 1):
#         plt.subplot(1, len(images), i)
#         plt.imshow(img, cmap='gray')
#         plt.title(title)
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

def process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels, display=True):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    denoised_images = [[func(noisy.copy()) for noisy in noisy_images] for func in denoise_funcs]

    if display:
        plot_results(original_image, noisy_images, denoised_images, noise_labels, denoise_labels)

    return noisy_images, denoised_images


# def process_image(image_path, noise_funcs, denoise_funcs, display=True):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    denoised_wavelet = [wavelet_denoising(noisy.copy()) for noisy in noisy_images]
    denoised_gaussian = [gaussian_denoise(noisy.copy()) for noisy in noisy_images]
    denoised_pm_f1 = [anisodiff_f1(noisy.copy(), 50, 0.1) for noisy in noisy_images]
    denoised_pm_f2 = [anisodiff_f2(noisy.copy(), 50, 0.1) for noisy in noisy_images]
    
    if display:
        plot_results(original_image, noisy_images, denoised_wavelet, denoised_gaussian, denoised_pm_f1, denoised_pm_f2, ['Speckle', 'Gaussian'])

    return noisy_images, denoised_wavelet, denoised_gaussian, denoised_pm_f1, denoised_pm_f2

def main():
    image_path = 'images/dog.jpg'
    noise_funcs = [lambda img: add_speckle_noise(img, var=0.05), lambda img: add_gaussian_noise(img, var=0.05)]
    process_image(image_path, noise_funcs, None)


# def process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels, display=True):
#     original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     noisy_images = [func(original_image.copy()) for func in noise_funcs]
#     denoised_images = [func(noisy) for noisy in noisy_images for func in denoise_funcs]
    
#     if display:
#         titles = ["Original"] + [label for label in noise_labels] + [f"{n_label} + {d_label}" for n_label in noise_labels for d_label in denoise_labels]
#         plot_results([original_image] + noisy_images + denoised_images, titles)
#     return noisy_images, denoised_images

def main():
    image_path = 'images/dog.jpg'
    noise_funcs = [
        lambda img: add_speckle_noise(img, var=0.05),
        lambda img: add_gaussian_noise(img, var=0.05)
    ]
    noise_labels = ['Speckle Noise', 'Gaussian Noise']
    
    denoise_funcs = [
        lambda img: wavelet_denoising(img, level=3),
        lambda img: gaussian_denoise(img, sigma=1.0),
        lambda img: anisodiff_f1(img, steps=50, K=0.1),
        lambda img: anisodiff_f2(img, steps=50, K=0.1)
    ]
    denoise_labels = [
        'Wavelet Denoising',
        'Gaussian Denoising',
        'Perona-Malik F1 Denoising',
        'Perona-Malik F2 Denoising'
    ]

    process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels)


# def main():
#     image_path = 'images/dog.jpg'
#     noise_funcs = [lambda img: add_speckle_noise(img, var=0.05), lambda img: add_gaussian_noise(img, var=0.05)]
#     process_image(image_path, noise_funcs, None)

#     # image_path = 'images/dog.jpg'
#     # noise_funcs = [
#     #     lambda img: add_speckle_noise(img, var=0.05),
#     #     lambda img: add_gaussian_noise(img, var=0.05)
#     # ]
#     # noise_labels = [
#     #     "Speckle Noise",
#     #     "Gaussian Noise"
#     # ]
#     # denoise_funcs = [
#     #     lambda img: wavelet_denoising(img, level=3),
#     #     lambda img: gaussian_denoise(img, sigma=1.0)
#     # ]
#     # denoise_labels = [
#     #     "Wavelet Denoise",
#     #     "Gaussian Denoise"
#     # ]

#     # noisy_images, denoised_images = process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels)

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