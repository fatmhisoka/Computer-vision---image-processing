from PyQt5.QtWidgets import  QFileDialog 
from PyQt5.QtGui import QPixmap, QImage
import cv2



def display_image( widget, image_):
    """
    Displays the given image on a Qt widget.

    Args:
        widget (QWidget): Qt widget to display the image.
        image (numpy.ndarray): Image to be displayed.

    Raises:
        ValueError: If the image shape is not supported.
    """
    
    image = image_.copy()
    height, width = image.shape[:2]  # Extract height and width

    if len(image.shape) == 3:  # Color image
        bytes_per_line = width * image.shape[2]  # Calculate bytes per line
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    elif len(image.shape) == 2:  # Grayscale image
        bytes_per_line = width  # Grayscale images have only one channel
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:
        raise ValueError("Unsupported image shape")


    pixmap = QPixmap.fromImage(q_image.rgbSwapped())  # Convert QImage to QPixmap (swapping BGR to RGB)
    resized_pixmap = pixmap.scaled(widget.size())
    # Set the resized pixmap on the QLabel
    widget.setPixmap(resized_pixmap)

def  browse(self):
    """
    Opens a file dialog for browsing and selecting an image file.

    Returns:
        numpy.ndarray: Loaded image as a numpy array, or None if no file is selected.
    """
    file_dialog = QFileDialog()
    filename, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
    if filename:
        return cv2.imread(filename)
