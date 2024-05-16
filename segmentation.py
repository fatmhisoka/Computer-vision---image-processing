import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage.segmentation import mark_boundaries
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import cv2

np.random.seed(42)


def kmeans(img, max_iter=100, K=2, threshold=0.85):
    """
    Apply K-Means clustering to an image to segment it into K colors.

    Parameters:
    - img: Input image (numpy array)
    - max_iter: Maximum number of iterations for K-Means algorithm (default: 100)
    - K: Number of clusters (default: 2)
    - threshold: Convergence threshold for centroid updates (default: 0.85)

    Returns:
    - segmented_image: Segmented image after applying K-Means clustering
    """



    # Change color to RGB (from BGR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flatten the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))

    # Convert pixel values to float type
    pixel_vals = np.float32(pixel_vals)

    # Handle NaN values in pixel values array by setting them to a small value
    pixel_vals[np.isnan(pixel_vals)] = 1e-6

    # Initialize cluster centroids randomly
    centroids = pixel_vals[np.random.choice(pixel_vals.shape[0], K, replace=False), :]

    # Handle NaN values in centroids array by setting them to a small value
    centroids[np.isnan(centroids)] = 1e-6

    # Initialize old centroids
    old_centroids = np.zeros_like(centroids)

    # Iteratively update centroids until convergence or maximum iterations reached
    for i in range(max_iter):
        # Assign each pixel to the nearest centroid
        distances = np.sqrt(((pixel_vals - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update cluster centroids
        for k in range(K):
            centroids[k] = np.mean(pixel_vals[labels == k], axis=0)

        # Handle NaN values in centroids array by setting them to a small value
        centroids[np.isnan(centroids)] = 1e-6

        # Check for convergence
        if np.abs(centroids - old_centroids).mean() < threshold:
            break
        old_centroids = centroids.copy()

    # Convert centroids to 8-bit values
    centers = np.uint8(centroids)

    # Assign each pixel to its corresponding cluster center
    segmented_data = centers[labels.flatten()]

    # Reshape segmented data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image


def region_growing(img, seed_point, threshold):
    """
    Apply region growing algorithm to an image from a seed point.

    Parameters:
    - img: Input image (numpy array)
    - seed_point: Seed point (tuple of x, y coordinates)
    - threshold: Threshold for similarity between pixels (float)

    Returns:
    - output_img: Image with region grown from the seed point
    """

    # Make a copy of the input image
    output_img = img.copy()

    # Get the dimensions of the image
    height, width, channels = img.shape

    # Initialize an empty image for output
    output = np.zeros_like(img, dtype=np.uint8)

    # Function to get neighboring pixels of a given point
    def get_neighbours(point):
        x, y = point
        neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbours if 0 <= n[0] < height and 0 <= n[1] < width]

    # Function to check similarity between two pixels
    def similarity(pixel1, pixel2, threshold):
        return np.sqrt(np.sum((pixel1 - pixel2)**2)) < threshold

    # Initialize a queue with the seed point
    queue = [seed_point]

    # Start region growing
    while queue:
        current_point = queue.pop(0)
        output[current_point] = (0, 255, 0)  # Set the pixel to green
        neighbours = get_neighbours(current_point)
        for neighbour in neighbours:
            # Check if the neighbor is not already visited and similar to the seed value
            if not np.any(output[neighbour]) and similarity(img[neighbour], img[seed_point], threshold):
                output[neighbour] = (0, 255, 0)  # Set the pixel to green
                queue.append(neighbour)

    # Create a mask for the green regions
    green_mask = np.all(output == (0, 255, 0), axis=-1)

    # Overlay the green regions onto the original image
    output_img[green_mask] = (0, 255, 0)

    return output_img

