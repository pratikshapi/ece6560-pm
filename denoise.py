import numpy as np
import pywt
import cv2

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

def anisodiff_f1(img, steps=10, K=0.1, del_t=0.25):
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

def anisodiff_f2(img, steps=50, K=4, del_t=0.25):
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
