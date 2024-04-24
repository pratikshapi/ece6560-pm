# metrics.py: Metrics calculations (psnr, mse)
import numpy as np
def psnr(target, ref):
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mse(target, ref):
    return np.mean((target - ref) ** 2)

def nmse(target, reference):
    l2_norm = np.linalg.norm(target - reference)
    nmse = l2_norm ** 2 / np.mean(reference ** 2)
    return nmse
