import numpy as np

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
