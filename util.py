import os
import cv2

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def create_directory_structure():
    root_dirs = ['high_contrast', 'low_contrast', 'normal']
    sub_dirs = ['varyingK', 'varyingDelt']
    noise_types = ['gauss', 'sp']

    base_path = 'denoised'
    for root in root_dirs:
        for sub in sub_dirs:
            for noise in noise_types:
                path = os.path.join(base_path, root, sub, noise)
                os.makedirs(path, exist_ok=True)

    # Additional directories for wavelet and gaussian + Perona-Malik denoising
    for root in root_dirs:
        for noise in noise_types:
            path = os.path.join(base_path, root, noise)
            os.makedirs(path, exist_ok=True)

