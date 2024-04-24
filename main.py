import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import logging

from noise import add_salt_pepper_noise, add_gaussian_noise
from denoise import wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2, F1, F2
from metrics import psnr, nmse, mse
from util import ensure_dir, create_directory_structure
from contrast import create_contrasted_noised_images

# Set up logging
logging.basicConfig(filename='denoise.log', level=logging.INFO)

plot_directory = 'plots'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)


def setup_logger():
    logger = logging.getLogger('PSNR_Results')
    if not logger.handlers:
        handler = logging.FileHandler('results.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()

def log_psnr_results(original_image, denoised_images, labels, logger):
    for denoised_image, label in zip(denoised_images, labels):
        psnr_value = psnr(original_image, denoised_image)
        mse_value = mse(original_image, denoised_image)
        logger.info(f"{label}: PSNR = {psnr_value}")


def optimize_pmf_parameters(image, K_values, step_counts, func):
    best_psnr, best_mse, best_K, best_steps = -np.inf, np.inf, None, None
    original_image = cv2.imread('images/original.jpg', cv2.IMREAD_GRAYSCALE)  # Path to the original image for PSNR comparison
    for K in K_values:
        for steps in step_counts:
            denoised_image = func(image, steps, K)
            current_psnr = psnr(original_image, denoised_image)
            current_mse = mse(original_image, denoised_image)
            if current_psnr > best_psnr:
                best_psnr, best_mse = current_psnr, current_mse
                best_K, best_steps = K, steps
    logging.info(f'best_K: {best_K}, best_steps: {best_steps}, best_psnr: {best_psnr}, best_mse: {best_mse}')
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

def process_image(original_image, noise_funcs, denoise_funcs):
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    denoised_images = []
    for noisy in noisy_images:
        denoised_images.extend([func(noisy.copy()) for func in denoise_funcs])
    return noisy_images, denoised_images

def ensure_dirs():
    dirs = ['noised', 'denoised', 'noised/high_contrast', 'noised/low_contrast', 'denoised/high_contrast', 'denoised/low_contrast']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def save_image(denoised_filename, best_image):
    directory = os.path.dirname(denoised_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    cv2.imwrite(denoised_filename, best_image)


def process_and_save_denoised_images(image_path, noise_type, contrast_type, denoise_func, param, param_value):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if 'varyingK' in denoise_func.__name__:
        denoised_image = denoise_func(image, K=param_value)
    elif 'varyingDelt' in denoise_func.__name__:
        denoised_image = denoise_func(image, del_t=param_value)

    directory = f'denoised/{contrast_type}/{denoise_func.__name__}/{noise_type}'
    filename = f'{param}_{param_value}.png'
    save_image(denoised_image, directory, filename)

def process_with_varying_k(noisy_image, noise_type, original_image):
    psnr_values_f1 = []
    psnr_values_f2 = []

    for K in K_values:
        denoised_f1 = anisodiff_f1(noisy_image.copy(), K=K, steps=10, del_t=0.1)
        denoised_f2 = anisodiff_f2(noisy_image.copy(), K=K, steps=50, del_t=0.1)
        
        f1_filename = f'denoised/varyingK/{noise_type}/f1_K_{K}.png'
        f2_filename = f'denoised/varyingK/{noise_type}/f2_K_{K}.png'
        
        save_image(f1_filename, denoised_f1)
        save_image(f2_filename, denoised_f2)
        
        psnr_f1 = psnr(denoised_f1, original_image)
        psnr_f2 = psnr(denoised_f2, original_image)
        
        logger.info(f"K = {K}: psnr_f1 = {psnr_f1}, psnr_f2 = {psnr_f2}")

        psnr_values_f1.append(psnr_f1)
        psnr_values_f2.append(psnr_f2)

    
    # Plotting the PSNR values
    plt.figure()
    plt.plot(K_values, psnr_values_f1, label='F1 Denoising', marker='o')
    plt.plot(K_values, psnr_values_f2, label='F2 Denoising', marker='o')
    plt.xlabel('K values')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR for Varying K with {noise_type} Noise')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
    plot_filename = f'{plot_directory}/{noise_type}_psnr_varying_k.png'
    plt.savefig(plot_filename)
    plt.close()



def process_with_varying_delt(noisy_image, noise_type, original_image):
    psnr_values_f1 = []
    psnr_values_f2 = []

    for del_t in del_t_values:
        denoised_f1 = anisodiff_f1(noisy_image.copy(), K=0.1, steps=10, del_t=del_t)
        denoised_f2 = anisodiff_f2(noisy_image.copy(), K=0.1, steps=50, del_t=del_t)
        
        f1_filename = f'denoised/varyingDelt/{noise_type}/f1_K_{del_t}.png'
        f2_filename = f'denoised/varyingDelt/{noise_type}/f2_K_{del_t}.png'
        
        save_image(f1_filename, denoised_f1)
        save_image(f2_filename, denoised_f2)

        psnr_f1 = psnr(denoised_f1, original_image)
        psnr_f2 = psnr(denoised_f2, original_image)

        logger.info(f"del_t = {del_t}: psnr_f1 = {psnr_f1}, psnr_f2 = {psnr_f2}")

        psnr_values_f1.append(psnr_f1)
        psnr_values_f2.append(psnr_f2)
    
    # Plotting the PSNR values
    plt.figure()
    plt.plot(del_t_values, psnr_values_f1, label='F1 Denoising', marker='o')
    plt.plot(del_t_values, psnr_values_f2, label='F2 Denoising', marker='o')
    plt.xlabel('delta t values')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR for Varying del_t with {noise_type} Noise')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
    plot_filename = f'{plot_directory}/{noise_type}_psnr_varying_del_t.png'
    plt.savefig(plot_filename)
    plt.close()


def process_with_varying_steps(noisy_image, noise_type, original_image):
    psnr_values_f1 = []
    psnr_values_f2 = []

    for steps in step_counts:
        denoised_f1 = anisodiff_f1(noisy_image.copy(), K=0.1, steps=steps, del_t=0.1)
        denoised_f2 = anisodiff_f2(noisy_image.copy(), K=0.1, steps=steps, del_t=0.1)
        
        f1_filename = f'denoised/varyingSteps/{noise_type}/f1_steps_{steps}.png'
        f2_filename = f'denoised/varyingSteps/{noise_type}/f2_steps_{steps}.png'
        
        save_image(f1_filename, denoised_f1)
        save_image(f2_filename, denoised_f2)
        
        psnr_f1 = psnr(original_image, denoised_f1)
        psnr_f2 = psnr(original_image, denoised_f2)

        logger.info(f"steps = {steps}: psnr_f1 = {psnr_f1}, psnr_f2 = {psnr_f2}")

        psnr_values_f1.append(psnr_f1)
        psnr_values_f2.append(psnr_f2)
    
    # Plotting the PSNR values
    plt.figure()
    plt.plot(step_counts, psnr_values_f1, label='F1 Denoising', marker='o')
    plt.plot(step_counts, psnr_values_f2, label='F2 Denoising', marker='o')
    plt.xlabel('Step counts')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR for Varying Steps with {noise_type} Noise')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
    plot_filename = f'plots/{noise_type}_psnr_varying_steps.png'
    plt.savefig(plot_filename)
    plt.close()


def plot_diffusion_coefficient_vs_gradient_line_and_save(F, equation_number, filename):
    image_gradients = np.linspace(0, 10, 100)
    plt.figure()

    for K in K_values:
        diffusion = F(image_gradients, K)
        plt.plot(image_gradients, diffusion, label=f'K = {K}')
    
    plt.title(f'Effect on diffusion by changing K')
    plt.xlabel('Image Gradient')
    plt.ylabel('Diffusion')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f'plots/{filename}_equation_{equation_number}.png')
    plt.close()  # Close the figure to free memory


def process_and_save_images(noisy_images, noise_labels, denoise_funcs, denoise_labels, base_directory="", K_values=None, step_counts=None):
    logger = setup_logger()  # Ensure the logger is set up once here

    for noisy, noise_label in zip(noisy_images, noise_labels):
        directory = f"{base_directory}{noise_label}/"
        
        for denoise_func, denoise_label in zip(denoise_funcs, denoise_labels):
            if 'perona_malik' in denoise_label:
                # Use the optimization function to get the best parameters and images
                best_K, best_steps, best_psnr, best_mse = optimize_pmf_parameters(noisy.copy(), K_values, step_counts, denoise_func)
                # Generate the best denoised image using the optimal parameters
                best_image = denoise_func(noisy.copy(), steps=best_steps, K=best_K)
                # Save the best denoised image
                denoised_filename = f'{directory}{noise_label}_with_{denoise_label}_K{best_K}_steps{best_steps}.png'
                save_image(denoised_filename, best_image)
                # Log the best PSNR and MSE values
                logger.info(f"{denoise_label} with K={best_K} and steps={best_steps}: Best PSNR = {best_psnr}, Best MSE = {best_mse}")
            else:
                # Process non-Perona-Malik methods directly
                denoised_image = denoise_func(noisy.copy())
                denoised_filename = f'{directory}{noise_label}_with_{denoise_label}.png'
                save_image(denoised_filename, denoised_image)
                # Calculate and log PSNR and MSE for direct methods
                psnr_value = psnr(original_image, denoised_image)
                mse_value = mse(original_image, denoised_image)
                logger.info(f"{denoise_label}: PSNR = {psnr_value}, MSE = {mse_value}")


def validate_image(image):
    if image is None or len(image.shape) != 2:
        raise ValueError("Invalid image. Ensure it is grayscale and exists.")

def generate_noised_images_if_absent(original_image_path, noise_funcs, noise_labels, contrast_levels):
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)
    
    # Check and create noised images for each contrast level
    for contrast in contrast_levels:
        for func, label in zip(noise_funcs, noise_labels):
            noised_image_path = f'noised/{contrast}/{label}.png'
            if not os.path.exists(noised_image_path):
                if contrast == 'normal':
                    noised_image = func(original_image.copy())
                else:
                    noised_image = create_contrasted_noised_images(original_image, contrast, func)
                save_image(noised_image, f'noised/{contrast}', f'{label}.png')

def generate_noisy_images(original_image, noise_funcs):
    noisy_images = [func(original_image.copy()) for func in noise_funcs]
    return noisy_images

def main():
    logger.info("Starting denoising process from main.py...")
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)
    base_directory = "denoised/"
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']
    generate_noised_images_if_absent(image_path, noise_funcs, noise_labels, ['normal'])
    denoise_funcs = [wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2]
    denoise_labels = ['wavelet_denoising', 'gaussian_denoising', 'perona_malik_f1_denoising', 'perona_malik_f2_denoising']

    noisy_images = generate_noisy_images(original_image, noise_funcs)
    process_and_save_images(noisy_images, noise_labels, denoise_funcs, denoise_labels, base_directory, K_values, step_counts)

def main_contrast():
    logger.info("Starting denoising process from main_contrast.py...")
    # Load pre-generated noisy images
    contrast_levels = ['high_contrast', 'low_contrast']
    noise_types = ['salt_pepper_noise', 'gaussian_noise']
    
    # Setup for denoising functions and labels
    denoise_funcs = [wavelet_denoising, gaussian_denoise, anisodiff_f1, anisodiff_f2]
    denoise_labels = ['wavelet_denoising', 'gaussian_denoising', 'perona_malik_f1_denoising', 'perona_malik_f2_denoising']

    # Process each contrast level
    for contrast in contrast_levels:
        noisy_images = []
        for noise_type in noise_types:
            image_path = f'noised/{contrast}/{noise_type}.png'
            noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            validate_image(noisy_image)
            noisy_images.append(noisy_image)

        noise_labels = [f'{contrast}_{type}' for type in noise_types]

        # Use a single call to process_and_save_images for each contrast level
        base_directory = f"denoised/{contrast}/"
        process_and_save_images(noisy_images, noise_labels, denoise_funcs, denoise_labels, base_directory, K_values, step_counts)

def main_varying_k():
    logger.info("Starting denoising process from main_varying_k.py...")
    original_image = cv2.imread('images/original.jpg', cv2.IMREAD_GRAYSCALE)  # Ensure this path is correct
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']

    generate_noised_images_if_absent(image_path, noise_funcs, noise_labels, ['normal'])

    gaussian_image = cv2.imread('noised/normal/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)
    salt_pepper_image = cv2.imread('noised/normal/salt_pepper_noise.png', cv2.IMREAD_GRAYSCALE)

    process_with_varying_k(gaussian_image, 'gaussian', original_image)
    process_with_varying_k(salt_pepper_image, 'salt_pepper', original_image)


def main_varing_delt():
    logger.info("Starting denoising process from main_varing_delt.py...")
    original_image = cv2.imread('images/original.jpg', cv2.IMREAD_GRAYSCALE)  
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']
    generate_noised_images_if_absent(image_path, noise_funcs, noise_labels, ['normal'])


    gaussian_image = cv2.imread('noised/normal/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)
    salt_pepper_image = cv2.imread('noised/normal/salt_pepper_noise.png', cv2.IMREAD_GRAYSCALE)

    process_with_varying_delt(gaussian_image, 'gaussian', original_image)
    process_with_varying_delt(salt_pepper_image, 'salt_pepper', original_image)

def main_varying_steps():
    logger.info("Starting denoising process with varying steps from main.py...")
    original_image = cv2.imread('images/original.jpg', cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)
    
    # Assuming images are already noised and saved under 'noised/normal'
    gaussian_image = cv2.imread('noised/normal/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)
    salt_pepper_image = cv2.imread('noised/normal/salt_pepper_noise.png', cv2.IMREAD_GRAYSCALE)

    process_with_varying_steps(gaussian_image, 'gaussian', original_image)
    process_with_varying_steps(salt_pepper_image, 'salt_pepper', original_image)


def generate_all_noised_images(original_image_path, noise_funcs, noise_labels, contrast_levels):
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(original_image)

    # Handle the normal contrast level first
    for func, label in zip(noise_funcs, noise_labels):
        noised_image_path = f'noised/normal/{label}.png'
        if not os.path.exists(noised_image_path):
            noised_image = func(original_image.copy())
            save_image(noised_image, 'noised/normal', f'{label}.png')

    # Now handle high and low contrast levels
    high_contrast, low_contrast = create_contrasted_noised_images(original_image)
    contrasts = {'high_contrast': high_contrast, 'low_contrast': low_contrast}
    for contrast_name, contrasted_image in contrasts.items():
        for func, label in zip(noise_funcs, noise_labels):
            noised_image_path = f'noised/{contrast_name}/{label}.png'
            if not os.path.exists(noised_image_path):
                noised_image = func(contrasted_image.copy())
                save_image(noised_image, f'noised/{contrast_name}', f'{label}.png')

def main_k_vs_gradient():
    plot_diffusion_coefficient_vs_gradient_line_and_save(F1, 3, 'F1')
    plot_diffusion_coefficient_vs_gradient_line_and_save(F2, 4, 'F2')


if __name__ == '__main__':

    image_path = 'images/dog.jpg'
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    K_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 10]
    del_t_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    step_counts = [5, 10, 50, 100, 150, 200]
    noise_funcs = [add_salt_pepper_noise, add_gaussian_noise]
    noise_labels = ['salt_pepper_noise', 'gaussian_noise']
    contrast_levels = ['high_contrast', 'low_contrast', 'normal']

    create_directory_structure()
    generate_all_noised_images(image_path, noise_funcs, noise_labels, contrast_levels)

    main()
    main_contrast()
    main_varying_k()
    main_varing_delt()
    main_varying_steps()
    main_k_vs_gradient()