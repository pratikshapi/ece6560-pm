import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

def madev(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(image, wavelet='db4', level=5):
    coeffs = pywt.wavedec2(image, wavelet, mode='per', level=level)
    sigma = (1/0.6745) * madev(coeffs[-1][0])
    uthresh = sigma * np.sqrt(2 * np.log(image.size))
    new_coeffs = [coeffs[0]] + [tuple(pywt.threshold(detail, value=uthresh, mode='hard') for detail in level) for level in coeffs[1:]]
    denoised_image = pywt.waverec2(new_coeffs, wavelet, mode='per')
    return denoised_image

def gaussian_denoise(image, kernel_size=5, sigma=1.5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def F1(dt, K):
    return np.exp(-1 * (np.power(dt, 2)) / (np.power(K, 2)))

def F2(dt, K):
    func = 1 / (1 + ((dt/K)**2))
    return func

def anisodiff_f1(image, steps=10, K=0.1, del_t=0.01, debug=False):
    denoised_image = np.zeros(image.shape, dtype=image.dtype)
    for i in range(steps):
        north_diff = image[:-2, 1:-1] - image[1:-1, 1:-1]
        south_diff = image[2:, 1:-1] - image[1:-1, 1:-1]
        east_diff = image[1:-1, 2:] - image[1:-1, 1:-1]
        west_diff = image[1:-1, :-2] - image[1:-1, 1:-1]
        denoised_image[1:-1, 1:-1] = image[1:-1, 1:-1] + del_t * (
            F1(north_diff, K) * north_diff + F1(south_diff, K) * south_diff +
            F1(east_diff, K) * east_diff + F1(west_diff, K) * west_diff)
        image = denoised_image.copy() 

        # Visual debugging
        if debug and i%10 == 0:
            plt.figure(figsize=(4, 4))
            plt.imshow(image, cmap='gray')
            plt.title(f'Step {i+1}')
            plt.axis('off')
            plt.show()
            
    return image

def anisodiff_f2(image, steps=50, K=4, del_t=0.01, debug=False):
    denoised_image = np.zeros(image.shape, dtype=image.dtype)
    for i in range(steps):
        north_diff = image[:-2, 1:-1] - image[1:-1, 1:-1]
        south_diff = image[2:, 1:-1] - image[1:-1, 1:-1]
        east_diff = image[1:-1, 2:] - image[1:-1, 1:-1]
        west_diff = image[1:-1, :-2] - image[1:-1, 1:-1]
        denoised_image[1:-1, 1:-1] = image[1:-1, 1:-1] + del_t * (
            F2(north_diff, K) * north_diff + F2(south_diff, K) * south_diff +
            F2(east_diff, K) * east_diff + F2(west_diff, K) * west_diff)
        image = denoised_image.copy() 

        # Visual debugging
        if debug and i%10 == 0:
            plt.figure(figsize=(4, 4))
            plt.imshow(image, cmap='gray')
            plt.title(f'Step {i+1}')
            plt.axis('off')
            plt.show()
            
    return image
