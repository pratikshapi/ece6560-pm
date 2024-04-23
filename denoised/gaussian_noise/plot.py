import matplotlib.pyplot as plt
import os

# Define the directory where images are stored
image_directory = '.'
image_filenames = os.listdir(image_directory)

# Define the substrings to search for in the filenames
substrings_order = ["original", "gaussian_noise.png", "gaussian_denoising", "wavelet_denoising", 
                    "perona_malik_f1_denoising", "perona_malik_f2_denoising"]

# Initialize a list to store file paths in the correct order
ordered_file_paths = []

# Sort the files according to the defined order in 'substrings_order'
for substr in substrings_order:
    for filename in image_filenames:
        if substr in filename:
            ordered_file_paths.append(os.path.join(image_directory, filename))

# Assuming we have 6 images to display in a 3x2 grid
fig, axes = plt.subplots(3, 2, figsize=(12, 8))

# Remove the gaps between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# List to keep track of the subplot indices
subplot_idx = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

# Display each image in its respective subplot
for idx, file_path in zip(subplot_idx, ordered_file_paths):
    # Read the image
    image = plt.imread(file_path)
    
    # Get the current subplot index
    i, j = idx
    
    # Show the image and remove axes
    axes[i, j].imshow(image, cmap='gray')
    axes[i, j].axis('off')
    
    # Optional: Set title (filename without extension)
    # axes[i, j].set_title(os.path.basename(file_path).split('.')[0], fontsize=8)
    axes[i, j].set_title(os.path.splitext(os.path.basename(file_path))[0], fontsize=8)

    

# Tight layout to maximize image display area
plt.tight_layout()

# Save the figure if needed
plt.savefig('normal_gauss_noise_plot.png', dpi=300)

# Show the plot
plt.show()

