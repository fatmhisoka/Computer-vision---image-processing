
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from collections import defaultdict
import numpy as np
import cv2 as cv
import math
from PIL import Image

import matplotlib.pyplot as plt

    

def hough_peaks(H, peaks, neighborhood_size=3):
    """Calculate the indices of the peaks.
    Args:
        H: the accumulator
        peaks: number of line peaks.
        neighborhood_size (int, optional): the size of the region to detect 1 line within. Defaults to 3.
    Returns:
        indices (np.ndarray): the indices of the peaks.
        H: accumulator that holds only the line points.
    """
    
    indices = []
    H1 = np.copy(H)  #copy the accumlaoter
    
    # loop through number of peaks to identify
    for i in range(peaks):
        idx = np.argmax(H1)  # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H to be 2d array
        indices.append(H1_idx)

        idx_y, idx_x = H1_idx  # separate x, y indices from argmax(H)
        
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (neighborhood_size / 2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (neighborhood_size / 2)
        if (idx_x + (neighborhood_size / 2) + 1) > H.shape[1]:
            max_x = H.shape[1]
        else:
            max_x = idx_x + (neighborhood_size / 2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (neighborhood_size / 2)) < 0:
            min_y = 0
        else:
            min_y = idx_y - (neighborhood_size / 2)
        if (idx_y + (neighborhood_size / 2) + 1) > H.shape[0]:
            max_y = H.shape[0]
        else:
            max_y = idx_y + (neighborhood_size / 2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if x == min_x or x == (max_x - 1):
                    H[y, x] = 255
                if y == min_y or y == (max_y - 1):
                    H[y, x] = 255

    # return the indices and the original Hough space with selected points
    return indices, H

def hough_lines_draw(img, indices, rhos, thetas):
    """Draw lines according to specific rho and theta
    Args:
        img (np.ndarray): image to draw on
        indices: indices of the peaks points
        rhos: norm distances of each line from origin
        thetas: the angles between the norms and the horizontal x axis
    """ 

    for i in range(len(indices)):
        # get lines from rhos and thetas
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # these are then scaled so that the lines go off the edges of the image
        y1 = int(y0 + 1000 * (a))  
        x1 = int(x0 + 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
     
def line_detection(image: np.ndarray,T_low,T_upper):
    """Fucntion that detects lines in hough domain
    Args:
        image (np.ndarray())
    Returns:
        accumulator: hough domain curves
        rhos: norm distances of each line from origin
        thetas: the angles between the norms and the horizontal x axis
    """
    
    grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurImg = cv.GaussianBlur(grayImg, (5,5), 1.5)
    edgeImg = cv.Canny(blurImg, T_low, T_upper)
    

    height, width = edgeImg.shape
    
    maxDist = int(np.around(np.sqrt(height**2 + width**2)))  # the max possiable line 
    
    thetas = np.deg2rad(np.arange(-90, 90))   
    rhos = np.linspace(-maxDist, maxDist, 2*maxDist)  
    
    accumulator = np.zeros((2 * maxDist, len(thetas)))  
    
    for y in range(height):
        for x in range(width):
            if edgeImg[y,x] > 0:   # check if it is edge
                for k in range(len(thetas)):  
                    r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(r) + maxDist, k] += 1
                    
    return accumulator, thetas, rhos

def hough_lines(T_low,T_high,neighborhood_size,source: np.ndarray,peaks: int = 10):
    """detect lines and draw them on the image
    Args:
        source (np.ndarray): image
        peaks (int, optional): number of line peaks. Defaults to 10.
    Returns:
        image: image with detected lines
    """
    
    src = np.copy(source)
    H, thetas, rhos = line_detection(src,T_low,T_high)
    indicies, H = hough_peaks(H, peaks, neighborhood_size) 
    hough_lines_draw(src, indicies, rhos, thetas)
    plt.imshow(src)
    plt.axis("off")
    return src
     
     
     
def find_hough_circles(image,  r_min, r_max, delta_r, num_thetas, bin_threshold, progress_bar, filter_size, min_edge_threshold , max_edge_threshold, post_process = False):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filterd_image = cv.GaussianBlur(gray_image, (filter_size, filter_size), 0)
    edge_image = cv.Canny(filterd_image, min_edge_threshold, max_edge_threshold)
    # cv.imshow("f",  edge_image)

    # Theta ranges
    dtheta = int(360 / num_thetas)

    ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
    thetas = np.arange(0, 360, step=dtheta)

    ## Radius ranges from r_min to r_max 
    rs = np.arange(r_min, r_max, step=delta_r)

    # Calculate Cos(theta) and Sin(theta) it will be required later
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    # Evaluate and keep ready the candidate circles dx and dy for different delta radius
    # based on the the parametric equation of circle.
    # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
    # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)
    circle_candidates = []
    for r in rs:
        for t in range(num_thetas):
            #instead of using pre-calculated cos and sin theta values you can calculate here itself by following
            #circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
            #but its better to pre-calculate and use it here.
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
  
    # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
    # aready present in the dictionary instead of throwing exception.
    accumulator = defaultdict(int)
    
    edge_pixels = np.argwhere(edge_image != 0)
        
        # Loop over edge pixels
    i=0 
    for y, x in edge_pixels:
        i+=1
        progress_bar.setValue(int(100 * i / len(edge_pixels)))
        
        print(i , len(edge_pixels))
            # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
        for r, rcos_t, rsin_t in circle_candidates:
            x_center = x - rcos_t
            y_center = y - rsin_t
            accumulator[(x_center, y_center, r)] += 1 #vote for current candidate

    # Output image with detected lines drawn
    output_img = image.copy()
    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
    out_circles = []
    
    # Sort the accumulator based on the votes for the candidate circles 
    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
            # Shortlist the circle for final result
            out_circles.append((x, y, r, current_vote_percentage))
            print(x, y, r, current_vote_percentage)
        else:
            break
      
  
  # Post process the results, can add more post processing later.
    if post_process :
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in out_circles:
            # Exclude circles that are too close of each other
            # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
            # Remove nearby duplicate circles based on pixel_threshold
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
            out_circles = postprocess_circles
    
    
    # Draw shortlisted circles on the output image
    for x, y, r, v in out_circles[:5]:
        output_img = cv.circle(output_img, (x,y), r, (0,255,0), 2)
        # winsound.Beep(440, 500)  # Beep with frequency 440 Hz and duration 500 ms
        
    return output_img, out_circles , 



def find_hough_ellipses(image,  a_min, a_max, delta_a, b_min, b_max, delta_b, num_thetas,filter_size , minedge , maxedge ,  bin_threshold,progress_bar,  post_process=False):
    
    # image preprosessing
    
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #convert to gray image
    gray_image = cv.GaussianBlur(gray_image,(filter_size , filter_size) ,0)  #blur the image
    edge_image = cv.Canny(gray_image, minedge, maxedge) #apply canny edged detection
    # cv.imshow("Detected Ellipses", edge_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # Theta ranges
    dtheta = int(360 / num_thetas) 
    thetas = np.arange(0, 360, step=dtheta)

    # Ranges for major and minor axes
    as_ = np.arange(a_min, a_max, step=delta_a)
    bs = np.arange(b_min, b_max, step=delta_b)

    # Calculate cos(theta) and sin(theta)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    # Evaluate and keep ready the candidate ellipses dx and dy for different major and minor axes
    # based on the parametric equation of ellipse.
    # x = x_center + a * cos(t) and y = y_center + b * sin(t),
    # where (x_center, y_center) is center of candidate ellipse with major axis a and minor axis b,
    # t is in range of [0, 2PI)
    ellipse_candidates = []
    for a in as_:
        for b in bs:
            for t in range(num_thetas):
                ellipse_candidates.append((a, b, int(a * cos_thetas[t]), int(b * sin_thetas[t])))

    # initlize Hough Accumulator
    accumulator = defaultdict(int)
    
    edge_pixels = np.argwhere(edge_image != 0)

    # Loop over edge pixels
    i = 0
    for y, x in edge_pixels:
        i += 1
        progress_bar.setValue(int(100 * i / len(edge_pixels)))
        print(i , len(edge_pixels))
        # Found an edge pixel, now find and vote for ellipse from the candidate ellipses passing through this pixel.
        for a, b, acos_t, bsin_t in ellipse_candidates:
            x_center = x - acos_t
            y_center = y - bsin_t
            accumulator[(x_center, y_center, a, b)] += 1  # vote for current candidate

    # Output image with detected ellipses drawn
    output_img = image.copy()
    # Output list of detected ellipses. A single ellipse would be a tuple of (xc, yc, a, b, threshold)

    out_ellipses = []
    # Sort the accumulator based on the votes for the candidate ellipses
    x = sorted(accumulator.items(), key=lambda i: -i[1])
    i=0
    for candidate_ellipse, votes in x:
        i+=1 
        x, y, a, b = candidate_ellipse
        current_vote_percentage = votes / num_thetas
        print(i , len(accumulator) , current_vote_percentage)
        if current_vote_percentage >= bin_threshold:
            # Shortlist the ellipse for final result
            out_ellipses.append((x, y, a, b, current_vote_percentage))
        else:
            break

    # Post process the results, if specified
    if post_process:
        pixel_threshold = 5
        postprocess_ellipses = []
        for xc, yc, a, b, v in out_ellipses:
            # Exclude ellipses that are too close to each other
            if all((xc - xcc) ** 2 + (yc - ycc) ** 2 > max(a, b) ** 2 for xcc, ycc, _, _, _ in postprocess_ellipses):
                postprocess_ellipses.append((xc, yc, a, b, v))
        out_ellipses = postprocess_ellipses

    # Draw shortlisted ellipses on the output image
    for xc, yc, a, b, _ in out_ellipses:
        output_img = cv.ellipse(output_img, (xc, yc), (a, b), 0, 0, 360, (0, 255, 0), 2)

    return output_img, out_ellipses






