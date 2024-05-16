
import os
import cv2
import numpy as np
import cv2 as cv
from random import randint
import math


class meanshift():
    def __init__(self, img,iter):
        self.result_image = np.zeros(img.shape,np.uint8)   #Initializes an output image (result_image) with the same size and data type
        self.radius = 90            #spatial radius set to 90. This represents the maximum distance a pixel can be from the seed point to be considered a neighbor.
        self.Iter = iter

    def getNeighbors(self,seed,matrix):
        """

        -This function takes a seed point (seed) and a feature matrix (matrix) as input.
        -This function essentially identifies pixels within the spatial radius of the seed point.
        -It iterates through each row in the matrix (representing a pixel)
         and calculates the Euclidean distance between the seed and that pixel.

        """
        neighbors = []
        for i in range(0,len(matrix)):
            Pixel = matrix[i]
            d = math.sqrt(sum((Pixel-seed)**2))
            if(d<self.radius):      #If the distance is less than the radius, the pixel's index is added to the neighbors list.
                 neighbors.append(i)
        return neighbors

    def markPixels(self,neighbors,mean,matrix):
        """

        -This function takes neighbors (neighbors), mean (mean), feature matrix (matrix), and cluster number (cluster) as input.
        -It iterates through the neighbors list (indices of pixels within spatial radius).
        -For each neighbor, it updates the corresponding location in the output image (result_image) with the mean value
        """
        for i in neighbors:
            Pixel = matrix[i]
            x=Pixel[3]
            y=Pixel[4]
            self.result_image[x][y] = np.array(mean[:3],np.uint8)
        return np.delete(matrix,neighbors,axis=0)   # removes the processed neighbors from the feature matrix (matrix)

    def calculateMean(self,neighbors,matrix):
        """

        -This function takes neighbors (neighbors) and feature matrix (matrix) as input.
        -It selects the corresponding rows from the feature matrix based on the provided neighbor indices.
        -It calculates the average for each feature (Red, Green, Blue, X-coordinate, Y-coordinate) using np.mean.
        -This function essentially computes the new mean (center) based on the neighboring pixels.

        """
        neighbors = matrix[neighbors]       #selects rows from the matrix based on the indices in the neighbors list.
        r=neighbors[:,:1]
        g=neighbors[:,1:2]
        b=neighbors[:,2:3]
        x=neighbors[:,3:4]
        y=neighbors[:,4:5]
        mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])

        return mean

    def createFeatureMatrix(self,img):
        """
        -This function takes an image (img) as input.
        -It iterates through each pixel in the image and creates a feature vector containing the pixel's color values (Red, Green, Blue) and its spatial coordinates (X, Y).
        -It converts the list of feature vectors into a NumPy array (F) for efficient processing.
        """
        h,w,d = img.shape
        F = []
        for row in range(0,h):
            for col in range(0,w):
                r,g,b = img[row][col]
                F.append([r,g,b,row,col])
        F = np.array(F)
        return F

    def performMeanShift(self,img):
        """

        This function performs the core mean shift segmentation process

        """

        F = self.createFeatureMatrix(img)   #creates a feature matrix
        while(len(F) > 0):
            randomIndex = randint(0,len(F)-1)       # It randomly selects an index from the feature matrix as the initial seed point
            seed = F[randomIndex]       #retrieves the feature vector (color and coordinates)
            initialMean = seed
            neighbors = self.getNeighbors(seed,F)   # finds neighboring pixels

            if(len(neighbors) == 1):
                F=self.markPixels([randomIndex],initialMean,F)      #if single pixel mark directly to the output and skip the iteration
                continue
            mean = self.calculateMean(neighbors,F)
            meanShift = abs(mean-initialMean)

            if(np.mean(meanShift)<self.Iter):
                F = self.markPixels(neighbors,mean,F)

        return self.result_image

# img=cv.imread("C:/Users/Mai M.Gamal/Desktop/pythonProject/computer_vision_tasks/images/Capture9.png")
# meanshift_obj = meanshift(img)
# result=meanshift_obj.performMeanShift(img)
# output_filename = "mean2.jpg"  # Change the file extension as needed
# output_path = os.path.join(os.getcwd(), output_filename)
# cv2.imwrite(output_path, result)