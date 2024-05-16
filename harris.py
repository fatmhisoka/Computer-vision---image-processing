from scipy import signal as sig
import numpy as np
from scipy import ndimage as ndi
import cv2
def apply_harris(img, k = 0.05, threshold = 0.01):
    """ 
    threshold: is the percentage of max of haris operator, it must be a float number between 0 and 1
    k: Sensitivity factor to separate corners from edges.
    Increasing k will make the algorithm more sensitive to corners but might also increase false positives.

    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    I_x = sig.convolve2d(img_gray, kernel_x, mode='same')
    I_y = sig.convolve2d(img_gray, kernel_y, mode='same')
    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

    # determinant
    detM = Ixx * Iyy - Ixy ** 2

    # trace
    traceM = Ixx + Iyy

    harrisR = detM - k * traceM ** 2
    img_copy = np.copy(img)
    img_copy[np.where(harrisR>threshold*harrisR.max())] = [255,0,255]
   
    return img_copy

def apply_harris_lambda(img, k=0.05, threshold=0.01, lambda_threshold=0.04):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    I_x = sig.convolve2d(img_gray, kernel_x, mode='same')
    I_y = sig.convolve2d(img_gray, kernel_y, mode='same')
    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

    # Calculate the Harris response
    detM = Ixx * Iyy - Ixy ** 2
    traceM = Ixx + Iyy
    harrisR = detM - k * traceM ** 2

    # Calculate the minimum eigenvalue
    lambda_min = 0.5 * (traceM - np.sqrt(traceM ** 2 - 4 * detM))

    # Apply λ- criterion
    harrisR[np.where(lambda_min < lambda_threshold)] = 0

    # Thresholding
    img_copy = np.copy(img)
    img_copy[np.where(harrisR > threshold * harrisR.max())] = [255, 0, 255]

    return img_copy