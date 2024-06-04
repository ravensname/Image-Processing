from mpi4py import MPI
import numpy as np
from math import exp
from PIL import Image
import time
import matplotlib.pyplot as plt
import numba
import os

# Define whether to use OpenMP or not
USE_OPENMP = True

if USE_OPENMP:
    os.environ["NUMBA_ENABLE_THREADING"] = "1"

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function for Gaussian kernel generation
def gaussian_cpu(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    m = kernel_size // 2
    n = kernel_size // 2
    gaussian_sum = 0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel_value = exp(-((i-m)**2 + (j-n)**2) / (2*sigma**2))
            kernel[i, j] = kernel_value
            gaussian_sum += kernel_value
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= gaussian_sum
    return kernel

# Function for convolution
@numba.njit(parallel=True)
def convolve_cpu(result, mask, image):
    image_rows, image_cols, _ = image.shape
    mask_rows, mask_cols = mask.shape
    
    delta_rows = mask_rows // 2
    delta_cols = mask_cols // 2
    
    for i in numba.prange(image_rows):
        for j in numba.prange(image_cols):
            s = np.zeros(3, dtype=np.float32)
            for k in range(mask_rows):
                for l in range(mask_cols):
                    i_k = i - k + delta_rows
                    j_l = j - l + delta_cols
                    if (i_k >= 0) and (i_k < image_rows) and (j_l >= 0) and (j_l < image_cols):
                        for c in range(3):  # Iterate over RGB channels
                            s[c] += mask[k, l] * image[i_k, j_l, c]
            for c in range(3):
                result[i, j, c] = s[c]

# Function to process a single image using CPU
def process_image_cpu(image, sigma, kernel_size):
    # Generate Gaussian kernel on CPU
    kernel = gaussian_cpu(sigma, kernel_size)
    
    # Perform convolution on CPU
    result = np.zeros_like(image)
    convolve_cpu(result, kernel, image)
    
    return result

# Function to load an image
def load_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    img_array = img_array.astype(np.uint8)  # Convert to uint8 dtype
    return img_array

# List of image paths
image_paths = ['image.jpg', 'image2.jpg']  # Adjust paths as needed

# Gaussian kernel parameters
sigma = 5.3
kernel_size = 10

# Distribute workload among MPI processes
images_per_process = len(image_paths) // size
start_index = rank * images_per_process
end_index = start_index + images_per_process
if rank == size - 1:
    end_index = len(image_paths)  # Last process may get extra images

# Process each image assigned to this MPI process
total_time_per_image_cpu = []
total_time_all_images_cpu = 0
start_total = time.time()

for i, image_path in enumerate(image_paths[start_index:end_index]):
    # Load image
    image = load_image(image_path)

    # Measure time for processing using CPU
    start_time_cpu = time.time()
    result_cpu = process_image_cpu(image, sigma, kernel_size)
    end_time_cpu = time.time()
    total_time_per_image_cpu.append(end_time_cpu - start_time_cpu)
    total_time_all_images_cpu += end_time_cpu - start_time_cpu

    # Save the original and processed images to files
    original_filename = f"original_{i}.png"
    processed_filename = f"processed_{i}.png"
    
    plt.imsave(original_filename, image)
    plt.imsave(processed_filename, result_cpu.astype(np.uint8))

# Gather results if necessary
# (e.g., if you want to collect all processed images on one process for visualization)

# Print total time per image and overall time for all images using CPU
print("\nTotal time per image (CPU) on process", rank, ":")
for i, time_per_image in enumerate(total_time_per_image_cpu):
    print("Image {}: {:.6f} seconds".format(i+1, time_per_image))
print("Overall time for all images (CPU) on process", rank, ": {:.6f} seconds".format(total_time_all_images_cpu))

# Gather timings from all processes
all_times = comm.gather(total_time_all_images_cpu, root=0)
if rank == 0:
    total_time_all_images = sum(all_times)
    print("\nOverall time for all images (CPU) across all processes: {:.6f} seconds".format(total_time_all_images))

# Display the saved plots after MPI processes have completed
comm.Barrier()  # Synchronize MPI processes before displaying plots

if rank == 0:
    for i in range(len(image_paths)):
        original_filename = f"original_{i}.png"
        processed_filename = f"processed_{i}.png"
        
        original_image = plt.imread(original_filename)
        processed_image = plt.imread(processed_filename)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(processed_image)
        plt.title("Processed Image (CPU)")
        plt.show()