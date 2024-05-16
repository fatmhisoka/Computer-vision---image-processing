import numpy as np
import numpy as np
from scipy.signal import convolve2d

from skimage.util import random_noise


def add_noise( image_orig , noise_type  ):
    """
    Adds noise to the original image based on the specified noise type.

    Args:
        image_orig (numpy.ndarray): Original image.
        noise_type (str): Type of noise to add ('uniform', 'gaussian', or 'salt & pepper').

    Returns:
        numpy.ndarray: Noisy image.
    """
    image = image_orig.copy()
    
    if noise_type=='uniform':
        noised_image = add_uniform_noise(image)
    elif noise_type == 'gaussian':
        noised_image = add_gaussien_noise(image)
    elif noise_type == 'salt & pepper':
        noised_image = add_salt_papper_noise(image)
    return noised_image

def add_uniform_noise(image , amplitude = 20):
    """
    Adds uniform noise to the image.

    Args:
        image (numpy.ndarray): Input image.
        amplitude (int, optional): Amplitude of the uniform noise. Defaults to 20.

    Returns:
        numpy.ndarray: Noisy image.
    """
    noise = np.random.uniform(-amplitude , amplitude , size = image.shape)
    noisy_image = image.astype(np.float32) + noise

    # noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_gaussien_noise( image):
    """
    Adds Gaussian noise to the image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Noisy image.
    """
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy_image = np.clip(image + gauss.astype(np.uint8) , 0 , 255)
    return noisy_image
    
def add_salt_papper_noise( image):
    """
    Adds salt and pepper noise to the image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Noisy image.
    """
    noisy_image = np.copy(image)

    salt_noise = np.random.rand(*image.shape[:2])
    noisy_image[salt_noise < .01] = 255
    pepper_noise = np.random.rand(*image.shape[:2])
    noisy_image[pepper_noise < .01] = 0

    return noisy_image
    