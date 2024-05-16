import glob
import math
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.image as mpimg


def get_filepaths(directory):
    '''getting the images paths'''

    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(filename)
            file_paths.append(filepath)
    return file_paths  # Self-explanatory.


def class_calc(images_paths, training_path):
    ''' to know the number of classes(faces)'''
    categories = []
    for image_path in images_paths:
        # "." which is the filename without extension
        # "_" face identifier
        category = (image_path.split(".")[0]).split("_")[1]
        if int(category) not in categories:
            categories.append(int(category))
    categories = sorted(categories)  # sorted in ascending order
    # assigned the length of the categories list, which represents the total number of classes (faces)
    class_num = len(categories)
    print(categories)
    return categories, class_num


def training(training_path):
    '''to get the training images and flattened array'''

    images_paths = get_filepaths(training_path)  # get paths
    height = 640
    width = 480
    training_images = np.ndarray(shape=(len(images_paths), height * width), dtype=np.float64)
    # read the images and get the 1D vector
    for i in range(len(images_paths)):
        path = training_path + '/' + images_paths[i]
        read_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(read_image, (width, height))
        training_images[i, :] = np.array(resized_image, dtype='float64').flatten()  # stored in row of training_images.

    print(training_images.shape)
    return training_images


def get_mean_normalized(training_path, training_images):
    images_paths = get_filepaths(training_path)  # training images
    height = 640
    width = 480

    ##Get Mean Face##
    """The mean is just the sum of all of the pictures divided by the number of pictures.
        As a result, we will have a “mean face”."""

    mean_face = np.zeros((1, height * width))
    for i in training_images:
        mean_face = np.add(mean_face, i)  # to sum all images
    # to get the mean face by dividing the summation by the length of images
    mean_face = np.divide(mean_face, float(len(images_paths))).flatten()

    ##Normailze Faces##
    """In summary, normalization is essential to ensure that the model focuses on
    the unique characteristics of each face by removing common features present in all images,
    such as lighting conditions and camera settings. It helps in improving the performance 
    and robustness of the facial recognition system."""

    normalised_training = np.ndarray(shape=(len(images_paths), height * width))
    for i in range(len(images_paths)):
        normalised_training[i] = np.subtract(training_images[i], mean_face)  # to substract mean face from each image

    return mean_face, normalised_training


def cov_mat(normalised_training):
    """calculating the covariance matrix allows us to understand how the pixel values
     of images change together across the training set"""

    cov_matrix = ((normalised_training).dot(normalised_training.T))  # dot product
    cov_matrix = np.divide(cov_matrix, float(len(normalised_training)))
    return cov_matrix


def eigen_val_vec(cov_matrix):
    '''
    -This function is designed to compute the eigenvalues and eigenvectors of a
     given covariance matrix, typically obtained from the normalized training set of    facial images.
    -The eigenvalues and eigenvectors are then sorted in (descending order) of importance based on the eigenvalues.
    -The eigenvectors are normalized to obtain the eigenfaces, which are the principal components of the facial images.'''

    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
    eigenfaces = preprocessing.normalize(eigvectors_sort)  # normalize eigen vectors

    return eigvalues_sort, eigenfaces


def get_reduced(eigenfaces, eigvalues_sort):
    '''to get the eigen faces till 90% '''

    var_comp_sum = np.cumsum(eigvalues_sort) / sum(eigvalues_sort)
    # eigen faces components
    reduced_data = []
    for i in (var_comp_sum):
        if i < 0.90:
            reduced_data.append(i)

    reduced_data = np.array(eigenfaces[:len(reduced_data)]).transpose()
    return reduced_data


def projected_data(training_images, reduced_data):
    """Training data :images, pixels
       reduced_data :eigenfaces, pixels"""

    proj_data = np.dot(training_images.transpose(), reduced_data)
    proj_data = proj_data.transpose()
    print(proj_data.shape)
    return proj_data


def weights(proj_data, normalised_training):

    w = np.array([np.dot(proj_data, i) for i in normalised_training])
    return w


def pca(unknown_face):
    training_path = "our_faces/data"
    height = 640
    width = 480

    # unknown_face = cv2.imread(path_unknown, cv2.IMREAD_GRAYSCALE)#read the image
    gray = cv2.cvtColor(unknown_face, cv2.COLOR_BGR2GRAY)
    unknown_face = cv2.resize(gray, (width, height))  # resize with 640*480 shape
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()  # get the flattened array
    training_images = training(training_path)
    mean_face, normalised_training = get_mean_normalized(training_path, training_images)
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)
    cov_matrix = cov_mat(normalised_training)
    eigvalues_sort, eigenfaces = eigen_val_vec(cov_matrix)
    reduced_data = get_reduced(eigenfaces, eigvalues_sort)
    proj_data = projected_data(training_images, reduced_data)
    w = weights(proj_data, normalised_training)
    w_unknown = np.dot(proj_data, normalised_uface_vector)  # get the weight of test face
    euclidean_distance = np.linalg.norm(w - w_unknown, axis=1)  # get the euclidean distance
    best_match = np.argmin(euclidean_distance)  # get the index of the best matched one
    output_image = training_images[best_match].reshape(640, 480)
    saved = mpimg.imsave('our_faces/FaceRecognized.png', output_image.reshape(640, 480), cmap="gray")


