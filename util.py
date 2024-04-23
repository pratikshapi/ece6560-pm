import os
import cv2

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
