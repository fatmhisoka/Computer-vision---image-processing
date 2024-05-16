import cv2
from cv2 import SIFT
import numpy as np
from random import randint
import time
import matplotlib.pyplot as plt
import os


def handle_matching(image1, image2, match_num, operation_flag):
    """
   takes two images as input and performs feature matching using SSD (Sum of Squared Differences) or NCC (Normal Cross Correlation).
    """

    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # convert images to gray to simplify and
    # speed up SIFT calculations
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # generate keypoints and descriptor using sift
    sift = cv2.SIFT_create()
    keypoints_1, descriptor1 = sift.detectAndCompute(img1_gray, None)
    # The keypoints: Locations of Interest list tells you where SIFT found potentially distinctive regions in the image.
    # The descriptor :Feature Descriptions array encodes the characteristics of those regions, allowing for comparison with features in another image
    keypoints_2, descriptor2 = sift.detectAndCompute(img2_gray, None)

    print("Key=",keypoints_2)
    print("--------------------------------------------")
    print("desc=",descriptor2)


    start = time.time()
    if operation_flag == 0:   #ssd or ncc
        matches = matching(descriptor1, descriptor2, calculate_ssd)
    else:
        matches = matching(descriptor1, descriptor2, calculate_ncc)

    matched_image = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2,
                                    matches[:match_num], image2, flags=2,matchesThickness=4)
    end = time.time()
    computational_time = end - start  # calculate computational time
    print("end")

    return computational_time, matched_image.astype(np.uint8)

def matching(descriptor1, descriptor2, calc_function):

    """
    This function iterates over all keypoints in the first image and finds the best match in the second image.

    descriptor1 and descriptor2: Descriptors of keypoints extracted from the two images.
    calc_function: A function that calculates the match score between two descriptors .
                    either sum of squared differences (SSD) or normalized cross correlations (ncc).
    """

    keypoints1 = descriptor1.shape[0]
    keypoints2 = descriptor2.shape[0]
    matches = []

    print("matching")

    for kp1 in range(keypoints1):                    # a loop over the keypoints in the first image

        distance = -np.inf                             #Sets the initial value of distance to negative infinity.
                                                       # This variable will store the highest similarity score

        second_image_index = -1                         #This variable will store the index of the keypoint in the second image
                                                        # that has the highest similarity to the current keypoint in the first image.


        for kp2 in range(keypoints2):

            value = calc_function(descriptor1[kp1], descriptor2[kp2])        #computes the similarity score between the descriptors.

            if value > distance:
                distance = value                    #Updates distance to the newly greater similarity score

                second_image_index = kp2            #Updates second_image_index to the index of the current keypoint
                                                    # in the second image with highest similarity .


        match = cv2.DMatch()                    # object represents a match between keypoints in two images.
        match.queryIdx = kp1                    #Sets the index of the keypoint in the query image (first image) to kp1
        match.trainIdx = second_image_index     #Sets the index of the keypoint in the train image (second image) to second_image_index
        match.distance = distance               #Sets the distance between the descriptors of the matched keypoints to distance
        matches.append(match)                   #Appends the cv2.DMatch object to the matches list


    matches = sorted(matches, key=lambda x: x.distance, reverse=True)   #objects sorted based on their distances in descending order,
                                                                        # matches with the highest similarity (smallest distance) will appear first in the list
    print("matching end")

    return matches


def calculate_ssd(descriptor1, descriptor2):
    """

    This function calculates the SSD between two descriptors.

        descriptor1 and descriptor2: Descriptors of keypoints from the two images.
        SSD Calculation:
        It iterates through each element of the descriptors and calculates the squared difference.
        SSD = Σ (descriptor1[m] - descriptor2[m])^2

    """
    ssd = 0
    for m in range(len(descriptor1)):
        ssd += (descriptor1[m] - descriptor2[m]) ** 2

    ssd =  -(np.sqrt(ssd))        #(-)sign to convert to similarity measure where higher values indicate stronger similarity.
    return ssd





def calculate_ncc(descriptor1, descriptor2):

    # Normalize descriptors
    out1_normalized = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1))
    out2_normalized = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2))
    # Compute element-wise multiplication
    correlation_vector = np.multiply(out1_normalized, out2_normalized)
    # Compute mean of the correlation vector
    correlation = float(np.mean(correlation_vector))
    return correlation



