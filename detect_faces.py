import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face


def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def convert_to_rgb(image):
    if image is None:
        raise ValueError("Input image is empty")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_and_draw_faces(image, face_cascade):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return image

