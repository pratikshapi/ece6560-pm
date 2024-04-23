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

def add_salt_pepper_noise(image, prob=0.05):
    output = np.copy(image)
    # Salt noise
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    output[tuple(coords)] = 255
    
    # Pepper noise
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    output[tuple(coords)] = 0    
    return output


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
    cols = num_noisy + 1  # For noisy images and the original image

    # First row: noisy images first, then the original image
    for i, img in enumerate(noisy, start=1):
        plt.subplot(5, cols, i)
        plt.imshow(img, cmap='gray')
        plt.title(noise_labels[i-1])
        plt.axis('off')

    # Place the original image last in the first row
    plt.subplot(5, cols, cols)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Subsequent rows: denoised images for each method
    row_offset = cols
    for d_index, denoised_group in enumerate(denoised):
        for i, img in enumerate(denoised_group, start=row_offset + 1):
            plt.subplot(5, cols, i)
            plt.imshow(img, cmap='gray')
            plt.title(f"{denoise_labels[d_index]}\n{noise_labels[(i - row_offset - 1) % num_noisy]}")
            plt.axis('off')
        row_offset += cols

    plt.tight_layout()
    plt.show()

def process_image(image_path, noise_funcs, noise_labels, denoise_funcs, denoise_labels, display=True):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    denoised_images = [[func(noisy.copy()) for noisy in noisy_images] for func in denoise_funcs]

    if display:
        plot_results(original_image, noisy_images, denoised_images, noise_labels, denoise_labels)

    return noisy_images, denoised_images

def main():
    image_path = 'images/dog.jpg'
    noise_funcs = [
        lambda img: add_salt_pepper_noise(img, prob=0.05),  # Replacing speckle noise
        lambda img: add_gaussian_noise(img, var=0.05)  # Assuming Gaussian noise remains the same
    ]
    noise_labels = ['Salt & Pepper Noise', 'Gaussian Noise']  # Update labels accordingly

    
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

if __name__ == '__main__':
    main()