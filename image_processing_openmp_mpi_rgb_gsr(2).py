from mpi4py import MPI
import numpy as np
from math import exp, radians, cos, sin
from PIL import Image
import time
import matplotlib.pyplot as plt
import multiprocessing
import mpi4py
import numba
import os

# Define whether to use OpenMP or not
USE_OPENMP = True

if USE_OPENMP:
    os.environ["NUMBA_ENABLE_THREADING"] = "4"

# https://mpi4py.readthedocs.io/en/stable/tutorial.html#running-python-scripts-with-mpi
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# https://numba.readthedocs.io/en/stable/user/parallel.html
# Function to scale an image
@numba.njit(parallel=True)
def scale_image(result, image, scale_factor):
    image_rows, image_cols, _ = image.shape
    new_rows = int(image_rows * scale_factor)
    new_cols = int(image_cols * scale_factor)
    for i in numba.prange(new_rows):
        for j in numba.prange(new_cols):
            i_new = min(int(i / scale_factor), image_rows - 1)
            j_new = min(int(j / scale_factor), image_cols - 1)
            for c in range(3):
                result[i, j, c] = image[i_new, j_new, c]

# Function to rotate an image
@numba.njit(parallel=True)
def rotate_image(result, image, angle):
    image_rows, image_cols, _ = image.shape
    angle_rad = radians(angle)
    center_i = image_rows / 2
    center_j = image_cols / 2
    for i in numba.prange(image_rows):
        for j in numba.prange(image_cols):
            new_i = (i - center_i) * cos(angle_rad) - (j - center_j) * sin(angle_rad) + center_i
            new_j = (i - center_i) * sin(angle_rad) + (j - center_j) * cos(angle_rad) + center_j
            if (new_i >= 0) and (new_i < image_rows) and (new_j >= 0) and (new_j < image_cols):
                for c in range(3):
                    result[i, j, c] = image[int(new_i), int(new_j), c]
            else:
                for c in range(3):
                    result[i, j, c] = 0

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
                        for c in range(3):
                            s[c] += mask[k, l] * image[i_k, j_l, c]
            for c in range(3):
                result[i, j, c] = s[c]

# Function to process a single image using CPU
def process_image_cpu(image, sigma, kernel_size):
    kernel = gaussian_cpu(sigma, kernel_size)
    
    result = np.zeros_like(image)
    convolve_cpu(result, kernel, image)
    
    return result

# Function to load an image
def load_image(image_path):
    return np.array(Image.open(image_path))

image_paths = ['image.jpg', 'image2.jpg', 'image3.jpg']

sigma = 5.3
kernel_size = 30

scale_factor = 0.8

rotation_angle = 45

# Distribute workload among MPI processes
images_per_process = len(image_paths) // size
start_index = rank * images_per_process
end_index = start_index + images_per_process
if rank == size - 1:
    end_index = len(image_paths) 

total_time_per_image = []
total_time_all_images = 0
start_total = time.time()
for image_path in image_paths[start_index:end_index]:
    image = load_image(image_path)
    
    start_time = time.time()
    
    result_blur = process_image_cpu(image, sigma, kernel_size)
    
    result_scaled = np.zeros_like(image)
    scale_image(result_scaled, image, scale_factor)

    result_rotated = np.zeros_like(image)
    rotate_image(result_rotated, image, rotation_angle)
    
    end_time = time.time()
    
    total_time_per_image.append(end_time - start_time)
    total_time_all_images += end_time - start_time

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    plt.subplot(1, 4, 2)
    plt.imshow(result_blur.astype(np.uint8))
    plt.title("Blurred Image")
    
    plt.subplot(1, 4, 3)
    plt.imshow(result_scaled.astype(np.uint8))
    plt.title("Scaled Image")
    
    plt.subplot(1, 4, 4)
    plt.imshow(result_rotated.astype(np.uint8))
    plt.title("Rotated Image")
    
    plt.show()

print("Total time per image:")
for i, time_per_image in enumerate(total_time_per_image):
    print("Image {}: {:.6f} seconds".format(i+1, time_per_image))
print("Overall time for all images: {:.6f} seconds".format(total_time_all_images))
