
import glob
import math
import matplotlib.image as mpimg

import cv2
import numpy as np
import os
import seaborn as snNew
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from tensorflow.python.keras.utils.np_utils import to_categorical



training_path = "our_faces/data"
test_path = "our_faces/test"




def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(filename)
            file_paths.append(filepath)
    labels = []
    for image_path in file_paths:
        label = (image_path.split(".")[0]).split("_")[1]
        labels.append(int(label))
    labels = np.array(labels)
    class_num = len(np.unique(labels))

    return np.array(file_paths), labels  # Self-explanatory.


def data_prep(direc, paths):
    height = 100
    width = 100
    images = np.ndarray(shape=(len(paths), height * width), dtype=np.float64)
    for i in range(len(paths)):
        path = direc + '/' + paths[i]
        read_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(read_image, (width, height))
        images[i, :] = np.array(resized_image, dtype='float64').flatten()


    return images



def accuracy(predictions, test_labels):
    """

    1-Takes predicted labels and actual labels as input.
    2-Calculates the accuracy (percentage of correct predictions) by dividing the number of correct predictions by the total number of samples.
    3-Returns the accuracy value.
    """
    l = len(test_labels)
    acc = sum([predictions[i] == test_labels[i] for i in range(l)]) / l
    return acc


def create_model():
    images_paths, labels = get_filepaths(training_path)
    test_paths, test_labels = get_filepaths(test_path)
    y = to_categorical(test_labels, num_classes=5)[:, 1:]
    print("y",y)
    training_images = data_prep(training_path, images_paths)
    test_images = data_prep(test_path, test_paths)
    model = RandomForestClassifier()
    model.fit(training_images, labels)

    prob_vector = model.predict_proba(test_images)
    prediction = model.predict(test_images)
    acc = accuracy(prediction, test_labels)

    print("probvec",prob_vector)
    return y, prob_vector,prediction,acc

def calc_fpr_tpr_thresh(class_num):

    test_paths, test_labels = get_filepaths(test_path)
    _,prob_vector,_,_=create_model()
    fpr = {}
    tpr = {}
    thresh = {}
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed

    # for i in range(1,5):
    fpr[class_num], tpr[class_num],  thresh[class_num] = roc_curve(test_labels, prob_vector[:, class_num-1], pos_label=class_num)
    # print( fpr[i], tpr[i], thresh[i],"/n")
    plt.plot(fpr[class_num], tpr[class_num], label=f'Class {class_num}')  # Plot each ROC curve with a different color
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('our_faces/ROC2.png')  # Save the ROC curve for all classes in one figure
    plt.close()

    return fpr,tpr,thresh




def calc_CM(probabilities, y_test,thresholds):
    for threshold in thresholds:
        threshold_vector = np.greater_equal(probabilities, threshold).astype(int)   #Creates a binary vector based on the threshold,
                                                                                    # indicating which probabilities are greater than or equal to the threshold for a single class.
        results = np.where(y_test == 1)[0]
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(threshold_vector)):
            if i in results:
                # which means that the actual value at these indices is 1
                if threshold_vector[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if threshold_vector[i] == 0:
                    tn += 1
                elif threshold_vector[i] == 1:
                    fp += 1


    cm = np.array([[fn, tp], [tn, fp]])
    return cm


def draw_CM(CM):
    DetaFrame_cm = pd.DataFrame(CM,  range(2),range(2))
    snNew.heatmap(DetaFrame_cm, annot=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    snNew.heatmap(DetaFrame_cm, annot=True, linewidths=.7, ax=ax)  # Sample figsize in inches

    plt.savefig('our_faces/CM.png')

#
# classnum=2
# y,prob_vector, _, _ = create_model()
# _,_,thresh=calc_fpr_tpr_thresh(classnum)
# # print("thr",thresh[3])
# # #
# CM= calc_CM(prob_vector[:,classnum-1], y[:,classnum-1],thresh[classnum])
# draw_CM(CM)
