
import itertools
from typing import Tuple
from PyQt5 import QtWidgets
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PyQt5.QtGui import QKeySequence, QPixmap, QImage
from filters import apply_gaussian_filter
#
import cv2
import math


class Snake:
    ####################Constant Parameters####################
    # min_distance_b_points = 5  # The minimum distance between two points to consider them overlaped
    # max_distance_b_points = 50  # The maximum distance to insert another point into the spline
    kernel_size_search = 7  # The size of search kernel

    ######################Variables#########################
    closed = True  # Indicates if the snake is closed or open.

    n_starting_points = 50  # The number of starting points of the snake.
    snake_length = 0
    # image = None        # The source image.
    gray = None  # The image in grayscale.
    binary = None  # The image in binary (threshold method).
    gradientX = None  # The gradient (sobel) of the image relative to x.
    gradientY = None  # The gradient (sobel) of the image relative to y.
    points = None

    ##############Define the constructor####################################
    def __init__(self, image,alpha=0.5,beta=0.5,gamma=0.5):
        # Sets the image and it's properties
        self.image = image

        self.alpha = alpha # The weight of the uniformity energy.
        self.beta =beta  # The weight of the curvature energy.
        self.gamma = gamma # The weight of the Image (gradient) Energy.

        # Image properties
        self.width = image.shape[1]
        self.height = image.shape[0]

        ######Set the line or curve to be closed loop ###################


        # Image variations used by the snake
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.gradientX = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=5)
        self.gradientY = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=5)

        ######To Draw our circle we need to define the center of the image As follow
        half_width = math.floor(self.width / 2)
        half_height = math.floor(self.height / 2)

        n = self.n_starting_points
        radius = half_width if half_width < half_height else half_height
        self.points = [np.array([
            half_width + math.floor(math.cos(2 * math.pi / n * x) * radius),
            half_height + math.floor(math.sin(2 * math.pi / n * x) * radius)])
            for x in range(0, n)
        ]

    def visuaize_Image(self):
        img = self.image.copy()

        # To draw lines between points, we have to define some parameters
        point_color = (0,255, 0)
        line_color = (0, 255, 0)
        thickness = 2  # Thickness of the lines and circles

        num_points = len(self.points)

        # Draw a line between the current and the next point
        for i in range(0, num_points - 1):
            cv2.line(img, tuple(self.points[i]), tuple(self.points[i + 1]), line_color, thickness)


        cv2.line(img, tuple(self.points[0]), tuple(self.points[num_points - 1]), line_color, thickness)


        return img




    ####### Normalization funtion to normalize the kernel of search
    def normalize(self,kernel):

        abs_sum = np.sum([abs(x) for x in kernel])
        return kernel / abs_sum if abs_sum != 0 else kernel

    def get_length(self): # calc 

        n_points = len(self.points)
        return np.sum([self.dist(self.points[i], self.points[(i + 1) % n_points]) for i in range(0, n_points)])

    #############Define the internal and external energy


    """
    The internal energy is responsible for:
        1. Forcing the contour to be continuous (E_cont)
        2. Forcing the contour to be smooth     (E_curv)
        3. Deciding if the snake wants to shrink/expand

    Internal Energy Equation:
        E_internal = E_cont + E_curv

    E_cont
        alpha * ||dc/ds||^2

        - Minimizing the first derivative.
        - The contour is approximated by N points P1, P2, ..., Pn.
        - The first derivative is approximated by a finite difference:

        E_cont = | (Vi+1 - Vi) | ^ 2
        E_cont = (Xi+1 - Xi)^2 + (Yi+1 - Yi)^2

    E_curv
        beta * ||d^2c / d^2s||^2

        - Minimizing the second derivative
        - We want to penalize if the curvature is too high
        - The curvature can be approximated by the following finite difference:

        E_curv = (Xi-1 - 2Xi + Xi+1)^2 + (Yi-1 - 2Yi + Yi+1)^2

    ==============================

    Alpha and Beta
        - Small alpha make the energy function insensitive to the amount of stretch
        - Big alpha increases the internal energy of the snake as it stretches more and more

        - Small beta causes snake to allow large curvature so that snake will curve into bends in the contour
        - Big beta leads to high price for curvature so snake prefers to be smooth and not curving

    :return:
    """



    # Continuity energy is the the external energy
    def dist(self,first_point, second_point):

        return np.sqrt(np.sum((first_point - second_point) ** 2))
    def cont_energy(self, p, prev):

        """

        E_cont
            alpha * ||dc/ds||^2

            - Minimizing the first derivative.
            - The contour is approximated by N points P1, P2, ..., Pn.
            - The first derivative is approximated by a finite difference:

            E_cont = | (Vi+1 - Vi) | ^ 2
            E_cont = (Xi+1 - Xi)^2 + (Yi+1 - Yi)^2

            at the following code we subtract the average distance from the actual distance and squaring the result
            helps quantify the smoothness of the contour, with larger deviations penalized more heavily,
            as reflected by the squared discrepancy. This approach encourages the contour to maintain
            a more uniform shape, as deviations from the average distance are minimized.

        """
        # The average distance between points in the snake
        avg_dist = self.snake_length / len(self.points)
        # The distance between the previous and the point being analysed
        un = self.dist(prev, p)
        dun = abs(un - avg_dist)

        return dun ** 2

    # Curveture energy
    def curv_energy(self, p, prev, next):

        """Compute the curvature energy at a given point on the contour.

        Parameters:
        - p: Current point (x, y) on the contour for which curvature energy is to be computed.
        - prev: Previous point (x_prev, y_prev) on the contour.
        - next: Next point (x_next, y_next) on the contour.

        Returns:
        - cn: Curvature energy at the current point.

        Description:
        The curvature energy (E_curv) penalizes high curvature along the contour, aiming to ensure smoothness.
        This energy term is calculated as the squared norm of the second derivative of the contour, scaled by a coefficient (beta).

        Formula:
        E_curv = beta * ||d^2c / d^2s||^2

        - d^2c / d^2s represents the second derivative of the contour with respect to the arc length.
        - beta is a coefficient that scales the contribution of curvature energy to the overall energy of the system.

        The curvature is approximated using finite differences between neighboring points on the contour:

        E_curv = (Xi-1 - 2Xi + Xi+1)^2 + (Yi-1 - 2Yi + Yi+1)^2

        Algorithm Steps:
        1. Compute the distances between the current point and the previous point (distant) and the next point (vn).
        2. Calculate the unit vectors in the x and y directions.
        3. Compute the curvature using the unit vectors.
        4. Return the curvature energy (cn) at the current point.

        Note:
        - If the distances between points are zero, curvature energy is set to zero to avoid division by zero errors.
        """


        # first the distance between the previous point and the current point
        dis_x = p[0] - prev[0]
        dis_y = p[1] - prev[1]
        distant = math.sqrt(dis_x ** 2 + dis_y ** 2)

        # Second: The distance between the currrent and next points
        vx = p[0] - next[0]
        vy = p[1] - next[1]
        vn = math.sqrt(vx ** 2 + vy ** 2)

        if distant == 0 or vn == 0:
            return 0

        cx = float(vx + dis_x) / (distant * vn)
        cy = float(vy + dis_y) / (distant * vn)
        cn = cx ** 2 + cy ** 2
        return cn

    """
        Implements the Active Contour (Snake) algorithm for image segmentation.
    
        Attributes:
        - gradientX: Gradient of the image in the x-direction.
        - gradientY: Gradient of the image in the y-direction.
        - width: Width of the image.
        - height: Height of the image.
        - alpha: Weight parameter for continuity energy.
        - beta: Weight parameter for curvature energy.
        - gamma: Weight parameter for gradient energy.
        - points: List of contour points representing the snake.
        - snake_length: Length of the snake contour.
        - kernel_size_search: Size of the search kernel for energy computation.
    
        Methods:
        - Grad_energy(p): Computes the gradient energy at a given point on the contour.
        - step(): Performs one iteration of the Active Contour algorithm.
    """
    def Grad_energy(self, p):
        """
               Computes the gradient energy at a given point on the contour.

               Parameters:
               - p: Current point (x, y) on the contour.

               Returns:
               - Gradient energy at the current point.

               Description:
               - If the current point lies outside the image boundaries, returns the maximum possible value of float64.
               - Otherwise, computes the gradient energy at the current point using the image gradients in both x and y directions.

               Formula:
               - Gradient Energy = -(GradientX^2 + GradientY^2)

               Note:
               - GradientX and GradientY are obtained from the sobel image gradients.
        """

        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
            # we use finfo function to get the objects are cached
            return np.finfo(np.float64).max

        return -(self.gradientX[p[1]][p[0]] ** 2 + self.gradientY[p[1]][p[0]] ** 2)


    def step(self):

        """
            Performs one iteration of the Active Contour algorithm.



            Description:
            - Initializes arrays to store energy values for continuity, curvature, and gradient.
            - Iterates through each point on the contour.
            - For each point, computes energy functions (continuity, curvature, and gradient) within a search kernel.
            - Normalizes the energy values.
            - Combines the normalized energy valsues using weights alpha, beta, and gamma.
            - Finds the minimum combined energy within the search kernel and updates the contour point accordingly.
            - Checks for boundary conditions and updates contour points.


        """
        self.snake_length = self.get_length()
        new_snake = self.points.copy()

        search_kernel_size = (self.kernel_size_search, self.kernel_size_search)
        kernel = math.floor(self.kernel_size_search / 2)  
        energy_cont = np.zeros(search_kernel_size)
        energy_curv = np.zeros(search_kernel_size)
        energy_grad = np.zeros(search_kernel_size)

        for i in range(0, len(self.points)):
            curr = self.points[i]
            prev = self.points[(i + len(self.points) - 1) % len(self.points)]
            next = self.points[(i + 1) % len(self.points)]

            for dx in range(-kernel, kernel):
                for dy in range(-kernel, kernel):
                    p = np.array([curr[0] + dx, curr[1] + dy])

                    # Calculates the energy functions on p
                    energy_cont[dx + kernel][dy + kernel] = self.cont_energy(p, prev)
                    energy_curv[dx + kernel][dy + kernel] = self.curv_energy(p, prev, next)
                    energy_grad[dx + kernel][dy + kernel] = self.Grad_energy(p)

            # Then, normalize the energies
            energy_cont = self.normalize(energy_cont)
            energy_curv = self.normalize(energy_curv)
            energy_grad = self.normalize(energy_grad)

            e_sum = self.alpha * energy_cont + self.beta * energy_curv + self.gamma * energy_grad
            emin = np.finfo(np.float64).max

            x, y = 0, 0
            for dx in range(-kernel, kernel):
                for dy in range(-kernel, kernel):
                    if e_sum[dx + kernel][dy + kernel] < emin:
                        emin = e_sum[dx + kernel][dy + kernel]
                        x = curr[0] + dx
                        y = curr[1] + dy

            # Boundary check
            x = 1 if x < 1 else x
            x = self.width - 2 if x >= self.width - 1 else x
            y = 1 if y < 1 else y
            y = self.height - 2 if y >= self.height - 1 else y

            new_snake[i] = np.array([x, y])

        self.points = new_snake


