import cv2
import numpy as np
from scipy.signal import convolve2d

from skimage.util import random_noise


def apply_edge(edge_type , original_image):
    """
    Applies the specified edge detection technique to the original image.

    Args:
        edge_type (str): Type of edge detection technique ('sobel', 'robert', 'prewitt', or 'canny').
        original_image (numpy.ndarray): Original input image.

    Returns:
        numpy.ndarray: Image with detected edges.
    """
    if edge_type == 'sobel':
        edged_img = sobel_edge(original_image)

    elif edge_type=='robert':
        edged_img =robert_edge(original_image)
    elif edge_type == 'prewitt':
        edged_img =prewitt_edge(original_image)
    elif edge_type == 'canny':
        edged_img =canny_edge(original_image)

    return edged_img

def sobel_edge(image):
    """
    Applies Sobel edge detection to the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with detected edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
    grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return gradient_magnitude

def robert_edge( img):
    """
    Applies Robert edge detection to the input image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with detected edges.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    robert_x = np.array([[1, 0, -1],
                            [0, 0, 0],
                            [-1, 0, 1]])
    robert_y = np.array([[0, -1, 0],
                            [1, 0, -1],
                            [0, 1, 0]])
    grad_x = cv2.filter2D(gray, cv2.CV_64F, robert_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, robert_y)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return gradient_magnitude

def prewitt_edge(img):
    """
    Applies Prewitt edge detection to the input image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with detected edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 1.5)
    prewitt_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
    grad_x = cv2.filter2D(gray, cv2.CV_64F, prewitt_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, prewitt_y)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return gradient_magnitude

def canny_edge( img):
    """
    Applies Canny edge detection to the input image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with detected edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 1.5)
    edges = cv2.Canny(gray, 0.05, 0.09)
    return edges



#canny edge from scratch task 2

import numpy as np
import cv2
from scipy.signal import convolve2d


def gaussian_kernel(self, size=5, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve2d(img, Kx)
    Iy = convolve2d(img, Ky)

    # G = np.hypot(Ix, Iy)
    G = np.sqrt(Ix ** 2 + Iy ** 2)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def non_max_suppression(img, D): # thining for edges (remove weak edges)
    #
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0   right left
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45    (digonal)
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90 baovw bleow
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135 other dia
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio=0.06, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(30)
    strong = np.int32(120)

    strong_i, strong_j = np.where(img >= highThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def hysteresis(img, weak=30, strong=120):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def findEdges(image_gray,size,sigma):
    img_smoothed = convolve2d(image_gray,  gaussian_kernel(size, sigma))
    gradientMat, thetaMat =  sobel_filters(img_smoothed)
    nonMaxImg =  non_max_suppression(gradientMat, thetaMat)
    thresholdImg =  threshold(nonMaxImg)
    img_final =  hysteresis(thresholdImg)
    return img_final