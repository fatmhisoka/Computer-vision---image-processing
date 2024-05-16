from scipy.ndimage import median_filter

import numpy as np
import cv2
import numpy as np
from scipy.signal import convolve2d


def apply_filter( noised_img , filter_size , filter_type):
    """
    Applies the specified filter to the input image.

    Args:
        noised_img (numpy.ndarray): Input image.
        filter_size (int): Size of the filter kernel.
        filter_type (str): Type of filter ('average', 'gaussian', or 'median').

    Returns:
        numpy.ndarray: Filtered image.
    """
    image = noised_img
    if filter_type == 'average':
        filterd_img = apply_avg_filter(image , filter_size)
    elif filter_type == 'gaussian':
        filterd_img = apply_gaussian_filter(image , filter_size)
    elif filter_type == 'median':
        filterd_img = apply_median_filter(image , filter_size)
    return filterd_img 

def apply_avg_filter( image , filter_size ):
    """
    Applies an average filter to the input image.

    Args:
        image (numpy.ndarray): Input image.
        filter_size (int): Size of the filter kernel.

    Returns:
        numpy.ndarray: Filtered image.
    """
    kernel_value = 1 / (filter_size * filter_size)
    kernel = np.full((filter_size, filter_size), kernel_value)
    filtered_image = np.zeros_like(image, dtype=np.float32)
    if len(image.shape) == 2:  # Grayscale image
        filtered_image = convolve2d(image, kernel, mode='same', boundary='symm')
    elif len(image.shape) == 3:  # Color image
        for i in range(image.shape[2]):  # Apply filter to each channel
            filtered_image[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same', boundary='symm')
    print("here" , filtered_image - cv2.blur(image, (filter_size, filter_size)))
    return filtered_image.astype(np.uint8)
    
def gaussian_kernel( sigma, size=3):
    """
    Generates a Gaussian kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        size (int, optional): Size of the kernel. Defaults to 3.

    Returns:
        numpy.ndarray: Gaussian kernel.
    """
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def apply_gaussian_filter( image,  kernel_size=3):
    """
    Applies a Gaussian filter to the input image.

    Args:
        image (numpy.ndarray): Input image.
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 3.

    Returns:
        numpy.ndarray: Filtered image.
    """
    sigma = 1
    kernel = gaussian_kernel(sigma, kernel_size)
    if len(image.shape) == 2:  # Grayscale image
        filtered_image = convolve2d(image, kernel, mode='same', boundary='symm')
    elif len(image.shape) == 3:  # Color image
        filtered_image = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):  # Apply filter to each channel
            filtered_image[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same', boundary='symm')
    else:
        raise ValueError("Unsupported image shape")

    return filtered_image.astype(np.uint8)


# def apply_median_filter( image , filter_size):
#     """
#     Applies a median filter to the input image.

#     Args:
#         image (numpy.ndarray): Input image.
#         filter_size (int): Size of the median filter kernel.

#     Returns:
#         numpy.ndarray: Filtered image.
#     """
#     test(image , filter_size)
#     '''
#     a test function to test my implemned filter 
#     '''
#     if len(image.shape) == 3:  # Color image
#         channels = [cv2.medianBlur(image[:, :, i], filter_size) for i in range(image.shape[2])]
#         return np.stack(channels, axis=2)
#     elif len(image.shape) == 2:  # Grayscale image
#         return cv2.medianBlur(image, filter_size)

def apply_median_filter(image, kernel_size):
    """
    Apply median filter to a colored image.

    Parameters:
    - color_img (numpy.ndarray): Input colored image (3D array) of shape (height, width, channels).
    - kernel_size (int): Size of the square median filter kernel. It must be an odd integer.

    Returns:
    - numpy.ndarray: Filtered colored image of the same shape as the input image.
    """
    # Pad the image to handle edges
    bd = kernel_size // 2
    median_img = np.zeros_like(image)

    for c in range(image.shape[2]):
        for i in range(bd, image.shape[0] - bd):

            for j in range(bd, image.shape[1] - bd):

                kernel = np.ravel(image[i - bd : i + bd + 1, j - bd : j + bd + 1, c])

                median = np.sort(kernel)[(kernel_size * kernel_size) // 2]

                median_img[i, j, c] = median

    return median_img


def test( image , kernel_size):
    if len(image.shape) == 3:  # Color image
        channels = [median_filter(image[:, :, i], size=kernel_size) for i in range(image.shape[2])]
        x =  np.stack(channels, axis=2)
    elif len(image.shape) == 2:  # Grayscale image
        x =  median_filter(image, size=kernel_size)
    
    
    
    
    
    
def low_pass(image, cutoff):
    """
    Applies a low-pass filter to the input image.

    Args:
        image (numpy.ndarray): Input image.
        cutoff (float): Cutoff frequency for the low-pass filter.

    Returns:
        numpy.ndarray: Low-pass filtered image.
    """
    return gaussian_blur(image, cutoff)

def high_pass(image, cutoff):
    """
    Generates a high-pass filtered image.

    Args:
        image (numpy.ndarray): Input image.
        cutoff (float): Cutoff frequency for the high-pass filter.

    Returns:
        numpy.ndarray: High-pass filtered image.
    """
    print("[{}]\tGenerating high pass image...".format(image))
    output=(image / 255) - low_pass(image, cutoff)
    return output

def gaussian_blur(image, sigma):
    """
    Applies Gaussian blur to the input image.

    Args:
        image (numpy.ndarray): Input image.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        numpy.ndarray: Blurred image.
    """
    print("[{}]\tCalculating Gaussian kernel...".format(image))
    size = 8 * sigma + 1
    if not size % 2:
        size = size + 1

    center = size // 2
    kernel = np.zeros((size, size))
    # Generate Gaussian blur.
    for y in range(size):
        for x in range(size):
            diff = (y - center) ** 2 + (x - center) ** 2
            kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    return convolution(image, kernel)
    
def convolution(img, kernel):
    """
    Performs convolution operation between the image and the kernel.

    Args:
        img (numpy.ndarray): Input image.
        kernel (numpy.ndarray): Convolution kernel.

    Returns:
        numpy.ndarray: Convolved image.
    """
    (image_h, image_w) = img.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    padded_kernel = np.zeros(img.shape[:2])
    start_h = (image_h - kernel_h) // 2
    start_w = (image_w - kernel_w) // 2
    padded_kernel[start_h: start_h + kernel_h, start_w: start_w + kernel_w] = kernel
    output = np.zeros(img.shape)
    for colour in range(3):
        Fi = np.fft.fft2(img[:, :, colour])
        Fk = np.fft.fft2(padded_kernel)
        output[:, :, colour] = np.fft.fftshift(np.fft.ifft2(Fi * Fk)) / 255

    return output
