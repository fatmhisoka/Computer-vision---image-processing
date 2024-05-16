import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import os
clusters_list = []
cluster = {}
centers = {}

def  Euclidean_distance(x1, x2):
    """
   This function calculates the Euclidean distance between two data points x1 and x2
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def clusters_average_distance(cluster1, cluster2):

    """
        1- calculates the average distance between two clusters.
        2- calculates the center of each cluster by taking the average of all data points within that cluster.
        3- calls the previously defined Euclidean_distance function to find the distance between the two cluster centers.
    """
    cluster1_center = np.average(cluster1)
    cluster2_center = np.average(cluster2)
    return  Euclidean_distance(cluster1_center, cluster2_center)


def initial_clusters(image_clusters):
    """

   Performs initial clustering of data points (pixels) into initial_k  clusters.

    """
    global initial_k    #define the initial number of clusters
    groups = {}         #Dictionary to store data points belonging to each cluster, identified by their average color.
    cluster_color = int(256 / initial_k)
    for i in range(initial_k):
        color = i * cluster_color
        groups[(color, color, color)] = []
    for i, p in enumerate(image_clusters):
        Cluster = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))    #finds the closest cluster center (represented by average color) in the groups dictionary
                                                                                # using min and a distance function.
        groups[Cluster].append(p)            #Data points are assigned to their closest clusters
    return [group for group in groups.values() if len(group) > 0]   #filter empty clusters


def get_cluster_center(point):
    """
   retrieves the cluster center for a given data point (point)
    """
    point_cluster_num = cluster[tuple(point)]   #Uses the cluster dictionary to find the cluster number associated with the data point
    center = centers[point_cluster_num]         #Retrieves the cluster center from the centers dictionary based on the cluster number.
    return center


def get_clusters(image_clusters):
    """
     This function performs the core agglomerative clustering process.
    """
    clusters_list = initial_clusters(image_clusters)

    while len(clusters_list) > clusters_number:
        cluster1, cluster2 = min(
            [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
            key=lambda c: clusters_average_distance(c[0], c[1]))        # find the pair of clusters (represented as cluster1 and cluster2) that are most similar based on their average distance.

        clusters_list = [cluster_itr for cluster_itr in clusters_list if
                         cluster_itr != cluster1 and cluster_itr != cluster2]   # filtered to remove the two clusters that were merged (cluster1 and cluster2).

        merged_cluster = cluster1 + cluster2

        clusters_list.append(merged_cluster)

    for cl_num, cl in enumerate(clusters_list): #assigning cluster num to each point
        for point in cl:
            cluster[tuple(point)] = cl_num

    for cl_num, cl in enumerate(clusters_list): #Computing cluster centers
        centers[cl_num] = np.average(cl, axis=0)


def apply_agglomerative_clustering( number_of_clusters, initial_number_of_clusters, image):
    global clusters_number
    global initial_k

    resized_image = cv2.resize(image, (256, 256))

    clusters_number = number_of_clusters
    initial_k = initial_number_of_clusters
    flattened_image = np.copy(resized_image.reshape((-1, 3)))

    get_clusters(flattened_image)
    output_image = []
    for row in resized_image:
        rows = []
        for col in row:
            rows.append(get_cluster_center(list(col)))
        output_image.append(rows)
    output_image = np.array(output_image, np.uint8)


    # output_filename = "Agg2.jpg"  # Change the file extension as needed
    # output_path = os.path.join(os.getcwd(), output_filename)
    # cv2.imwrite(output_path, output_image)

    return output_image





# apply_agglomerative_clustering( number_of_clusters=5, initial_number_of_clusters=20, image =cv2.imread("C:/Users/Mai M.Gamal/Desktop/pythonProject/computer_vision_tasks/images/Capture9.png"))