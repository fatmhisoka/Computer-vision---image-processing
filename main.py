from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import  QApplication, QMainWindow, QShortcut
from PyQt5.QtGui import QKeySequence, QPixmap, QImage
import sys
from PyQt5.QtCore import QCoreApplication

from scipy import signal as sig
import numpy as np
from scipy import ndimage as ndi
import cv2
import numpy as np
import cv2
import os
from mainwindow import Ui_MainWindow
import snake as sn
from PyQt5 import QtWidgets
from scipy.ndimage import gaussian_filter,sobel
from scipy import ndimage as filt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from filters import*
from load_and_display import*
from noise import*
from edges import*
from hough import *
from normalize_equalize_threshold import*
from segmentation import *
from my_sift import MY_SIFT
from harris import apply_harris
from Thresholding import*
from meanshift import*
from agglmorative import*
import time
import normalize_equalize_threshold as th
from spectral_thresholding import *
from  luv import*
from image_matching import*
from detect_faces import *
from pca import *
from ROC import*

class MyWindow(QMainWindow):   
    
    def __init__(self ):
        super(MyWindow , self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  
        self.setMinimumSize(800, 600)

        self.original_image=None
        self.noisy_image=None
        self.before_low_image=None
        self.before_high_image=None
        self.before_threshold=None
        self.before_conversion=None
        self.before_edge=None

        ####task 2###
        self.snake=None
        self.canny_image=None
        self.line_image = None


        ###task3####
        self.match_1=None
        self.match_2=None

        self.input_bgr = 2 * [0]
        self.input_rgb = 2 * [0]
        self.harris_input = None
        self.harris_output = None
        
        self.sift_img1 = None
        self.sift_img2 = None


        ###task4####
        self.local_thresholding_image = None
        self.mean_shift_image = None
        self.agglmorative_image = None
        self.spec_img = None
        self.luv_img = None
        
        
        # self.ui.out_img__reg.setMouseTracking(True)  # Enable mouse tracking
        self.ui.orig_img_reg.mouseDoubleClickEvent = self.apply_regoin
        






        ''' here write all the connetions'''
        QShortcut(QKeySequence("ctrl+o"), self).activated.connect(self.load_image)
        self.ui.action_open_img.triggered.connect(self.load_image)

        ## tab 1 connections
        self.ui.combo_filter.currentIndexChanged.connect(lambda: self.handle_combo(self.ui.combo_filter))
        self.ui.combo_noise.currentIndexChanged.connect(lambda: self.handle_combo(self.ui.combo_noise))
        self.ui.combo_gray_color.currentIndexChanged.connect(lambda: self.handle_combo(self.ui.combo_gray_color))
        self.ui.btn_filter_Apply.clicked.connect(self.apply_and_dispaly_filter)
        self.ui.combo_edge.currentIndexChanged.connect(self.handle_edge)

        ## tab 2 connections
        self.ui.slider_1.valueChanged.connect(lambda: self.thresholding(self.before_threshold))
        self.ui.slider_2.valueChanged.connect(lambda: self.thresholding(self.before_threshold))
        self.ui.slider_3.valueChanged.connect(lambda: self.thresholding(self.before_threshold))
        self.ui.slider_4.valueChanged.connect(lambda: self.thresholding(self.before_threshold))
        self.ui.combo_thersholding.currentIndexChanged.connect(lambda: self.thresholding(self.before_threshold))
        self.ui.comboBox.currentIndexChanged.connect(self.convert)
        
        ## tab 3 connections
        self.ui.btn_browse_1.clicked.connect(lambda :self.browse(0))
        self.ui.btn_browse_2 .clicked.connect(lambda :self.browse(1))
        self.ui.spin_lp_cuttoff.setValue(1)
        self.ui.spin_hp_cuttoff.setValue(1)
        self.ui.spin_lp_cuttoff.valueChanged.connect(lambda :self.handle_spinbox(0))
        self.ui.spin_hp_cuttoff.valueChanged.connect(lambda :self.handle_spinbox(1))
        self.ui.apply.clicked.connect(self.hybrid_image)


        ### task 2 ###
        self.ui.apply_snake.clicked.connect(lambda: self.active_contour(self.snake))
        ### canny tab
        self.ui.canny_Apply.clicked.connect(lambda : self.canny_detection(self.canny_image))
        self.ui.line_Apply.clicked.connect(lambda : self.hough_line(self.line_image))
        self.ui.comboBox_color_line.currentIndexChanged.connect(self.hough_line)
        self.ui.Lth_Slider.setMinimum(1)
        self.ui.Lth_Slider.setMaximum(400)
        self.ui.Lth_Slider.setSingleStep(10)
        self.ui.spinBox__line.setMinimum(1)
        self.ui.spinBox__line.setMaximum(300)

        self.ui.Hth_Slider.setMinimum(10)
        self.ui.Hth_Slider.setMaximum(500)
        self.ui.Hth_Slider.setSingleStep(10)


        self.ui.neigh_Slider.setMinimum(1)
        self.ui.neigh_Slider.setMaximum(200)
        self.ui.neigh_Slider.setSingleStep(5)
        
        
        #cirle tab connections
        self.ui.btn_apply_cir_detect.clicked.connect(self.apply_circle)
        self.ui.minR_Slider.valueChanged.connect(lambda value: self.ui.label_Min.setText("Min R " + str(value)))
        self.ui.maxR_Slider.valueChanged.connect(lambda value: self.ui.label_max.setText("Max R " + str(value)))
        self.ui.deltaR_Slider.valueChanged.connect(lambda value: self.ui.label_deltaR.setText("delta R " + str(value)))
        self.ui.bin_Slider.valueChanged.connect(lambda value: self.ui.label_bin.setText("Bin Threshold " + str(value)))
        self.ui.num_thetas_Slider.valueChanged.connect(lambda value: self.ui.labelnum_thetas.setText("num of thetas " + str(value)))
        self.ui.min_edge_threshold_Slider.valueChanged.connect(lambda value: self.ui.label_min_edge_threshold.setText("min edge th " + str(value)))
        self.ui.max_edge_threshold_Slider.valueChanged.connect(lambda value: self.ui.label_max_edge_threshold.setText("max edge th " + str(value)))
        self.ui.filterSize_Slider.valueChanged.connect(lambda value: self.ui.label_filter_size.setText("filter size " + str(value)))

        #ellipse tab connections
        self.ui.apply_ellipse.clicked.connect(self.apply_ellipse)



        #######match tab connections#####

        self.ui.browse_match_1.clicked.connect(lambda :self.browse(0))
        self.ui.browse_match_2.clicked.connect(lambda :self.browse(1))
        self.ui.matching_apply.clicked.connect(lambda :self.match(self.match_1,self.match_2))
        self.ui.harris_Apply.clicked.connect(self.harris_match)

        ############sift tab connections ########
        self.ui.sift_browse_img_1.clicked.connect(lambda :self.browse(0))
        self.ui.sift_browse_img_2.clicked.connect(lambda :self.browse(1))
        self.ui.sift_Apply.clicked.connect(self.apply_sift)

        ############ Thresholding ########
        self.ui.apply_opt.clicked.connect(self.optimal_thresholding)
        self.ui.apply_otsu.clicked.connect(self.otsu_thresholding)
        self.ui.apply_manual.clicked.connect(self.manual_thresholding)
        self.ui.apply_spec.clicked.connect(self.spectral_thresholding)

        self.ui.slider_6.valueChanged.connect(lambda: self.manual_thresholding(self.image_threshold))
        self.ui.slider_5.valueChanged.connect(lambda: self.manual_thresholding(self.image_threshold))
        self.ui.slider_7.valueChanged.connect(lambda: self.manual_thresholding(self.image_threshold))
        self.ui.slider_8.valueChanged.connect(lambda: self.manual_thresholding(self.image_threshold))
        self.ui.comboBox_2.currentIndexChanged.connect(lambda: self.manual_thresholding(self.image_threshold))

        ############### Mean Shift ###############33
        self.ui.apply_mean_shift.clicked.connect(lambda :self.mean_shift(self.mean_shift_image))
        self.ui.apply_agg.clicked.connect(lambda :self.agglomrative(self.agglmorative_image))
        self.ui.apply_luv.clicked.connect(lambda :self.luv_map(self.luv_img))

        
        
        # self.ui.apply_reg.clicked.connect(self.apply_regoin)
        self.ui.apply_k.clicked.connect(self.apply_kmeans)
        self.ui.detect_Apply.clicked.connect(self.apply_detect)
        self.ui.recogn_apply.clicked.connect(self.Recognize)


        ###################33
        self.ui.view.clicked.connect(self.roc)


  
    def load_image(self):
        '''this function is called when the user want to chose and display an
        image to add noise and apply filters and also diplay edgs'''
        self.original_image = browse(self)
        if self.ui.tabWidget_2.currentIndex()==0:    # Task 1
            if self.ui.tabWidget.currentIndex()==0:    

                display_image(self.ui.orig_img_1,  self.original_image)
                self.handle_edge()
                self.add_and_display_noise()
                self.apply_and_dispaly_filter()

            if self.ui.tabWidget.currentIndex()==1:
                display_image(self.ui.orig_img_2, self.original_image)
                self.equalization(self.original_image)
                self.normalize(self.original_image)
                self.before_threshold=self.original_image
                self.before_conversion = self.original_image
                self.ui.comboBox.setCurrentIndex(0)

        if self.ui.tabWidget_2.currentIndex() == 1: # Task 2
            if self.ui.tabwidget.currentIndex() == 0:
                display_image(self.ui.orig_img_canny, self.original_image)
                self.canny_image=np.copy(self.original_image)
                self.canny_detection(self.canny_image)

            if self.ui.tabwidget.currentIndex() == 1:
                display_image(self.ui.orig_img_line, self.original_image)
                self.ui.comboBox_color_line.setCurrentIndex(0)

                self.line_image = np.copy(self.original_image)
                self.hough_line( self.line_image)
                #self.ui.comboBox_color_line.setCurrentIndex(0)

            if self.ui.tabwidget.currentIndex() == 2:
                display_image(self.ui.orig_img_ellipse, self.original_image)
                # self.apply_ellipse()
                #self.ui.comboBox_color_line.setCurrentIndex(0)

            if self.ui.tabwidget.currentIndex() == 3:
                display_image(self.ui.orig_img_circle, self.original_image)

            if self.ui.tabwidget.currentIndex() == 4:
                display_image(self.ui.orig_img_snake, self.original_image)
                self.snake = np.copy(self.original_image)
                self.active_contour(self.snake)

        if self.ui.tabWidget_2.currentIndex() == 2:  # Task 3
            if self.ui.tabwidget_3.currentIndex() == 0:
                display_image(self.ui.orig_img_harris, self.original_image)
                self.img_bgr = np.copy(self.original_image)
                self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
                self.harris_input = self.img_bgr

        if self.ui.tabWidget_2.currentIndex() == 3: # Task4
            if self.ui.tabWidget_4.currentIndex() == 0: # thersholding_tab
                if self.ui.thresholding.currentIndex() == 0:
                    display_image(self.ui.orig_img_optimal, self.original_image)
                if self.ui.thresholding.currentIndex() == 1:
                    display_image(self.ui.orig_img_otsu, self.original_image)
                if self.ui.thresholding.currentIndex() == 2:
                    self.image_threshold=self.original_image
                    display_image(self.ui.orig_img_manual, self.original_image)
                if self.ui.thresholding.currentIndex() == 3:
                    self.spec_img=self.original_image
                    display_image(self.ui.orig_img_spec, self.original_image)
            if self.ui.tabWidget_4.currentIndex() == 1:  #cluster_tab
                if self.ui.clustering.currentIndex() == 0: #k means tab
                    display_image(self.ui.orig_img_k, self.original_image)
                if self.ui.clustering.currentIndex() == 1: 
                    self.mean_shift_image=self.original_image
                    display_image(self.ui.orig_img_shift, self.original_image)
                if self.ui.clustering.currentIndex() == 2: # region tab
                    display_image(self.ui.orig_img_reg, self.original_image)
                if self.ui.clustering.currentIndex() == 3:
                    self.agglmorative_image=self.original_image
                    display_image(self.ui.orig_img_agg,self.original_image)
                if self.ui.clustering.currentIndex() == 4:
                    self.luv_img=self.original_image
                    display_image(self.ui.orig_img_luv,self.original_image)

        if self.ui.tabWidget_2.currentIndex() == 4: # Task5
            if self.ui.tabwidget_5.currentIndex() == 0:
                display_image(self.ui.orig_img_detect, self.original_image)
            if self.ui.tabwidget_5.currentIndex() == 1:
                display_image(self.ui.orig_img_recogn, self.original_image)




    ###### tab 1  ######
    def handle_combo(self ,widget):
        """
        Handle events related to combo box changes.

        Args:
            widget: Combo box widget triggering the event.
        """
        if self.original_image is not None:
            if widget == self.ui.combo_noise:
                self.add_and_display_noise()
            elif widget == self.ui.combo_filter:
                self.apply_and_dispaly_filter()
            elif widget == self.ui.combo_edge:
                self.handle_edge()

    def add_and_display_noise(self ):
        """
        Add noise to the original image and display the noisy image.
        """
        self.noisy_image = add_noise( self.original_image , self.ui.combo_noise.currentText())
        display_image(self.ui.noise_img , self.noisy_image)
        self.apply_and_dispaly_filter()
    
    def apply_and_dispaly_filter(self ):
        """
        Apply selected filter to the noisy image and display the filtered image.
        """
        self.filterd_image = apply_filter( self.noisy_image  , self.ui.spin_filter_size.value() , self.ui.combo_filter.currentText())
        display_image(self.ui.filtre_img , self.filterd_image)

    def handle_edge(self):
        """
        Handle events related to edge detection.
        """
        edged_image = apply_edge(self.ui.combo_edge.currentText() , self.original_image)
        display_image(self.ui.edge_img, edged_image)
   
    ###### tab 2  ######
    def normalize(self,img):
        """
        Normalize the image and display it.

        Args:
            img: Input image.
        """
        norm_img=normalize(img)
        display_image(self.ui.norm_img, norm_img)

        self.draw_histogram(norm_img,1)
        self.draw_distribution(norm_img,1)

    def equalization(self,img):
        """
        Perform histogram equalization on the image and display it.

        Args:
            img: Input image.
        """
        equalized_img=equalization(img)

        display_image(self.ui.equlized_img, equalized_img)


        self.draw_histogram(equalized_img,0)
        self.draw_distribution(equalized_img,0)
        self.rgb_histogram(equalized_img)

    def rgb_histogram(self, image):
        """
        Plot RGB histograms for the given image.

        Args:
            image: Input image.
        """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_r = image[:, :, 0]
        img_g = image[:, :, 1]
        img_b = image[:, :, 2]  # separating the R, G, and B channels into different arrays
        # Converting the 2-d arrays to 1-d arrays
        img_r = img_r.flatten()
        img_g = img_g.flatten()
        img_b = img_b.flatten()

        plt.figure()
        fig, ax = plt.subplots(1, 3)  # creating figure
        bins = np.arange(256)
        # Ploting the histograms for the data
        ax[0].hist(img_r, bins=bins, color='r')
        ax[1].hist(img_g, bins=bins, color='g')
        ax[2].hist(img_b, bins=bins, color='b')
        plt.title('Histogram')
        output_folder = os.getcwd()
        histogram_path = os.path.join(output_folder, 'rgb_histogram.png')
        plt.savefig(histogram_path)
        pixmap = QPixmap("rgb_histogram.png")
        resized_pixmap = pixmap.scaled(self.ui.equalize_histo.size())
        self.ui.rgb_histo.setPixmap(resized_pixmap)

        # display_image(self.ui.rgb_histo, fig.astype(np.uint8))


        fig1, ax1 = plt.subplots(1, 2)
        ax1[0].hist(img_r, bins=bins, cumulative=True, histtype="step", label="red", color="r")
        ax1[0].hist(img_g, bins=bins, cumulative=True, histtype="step", label="green", color="g")
        ax1[0].hist(img_b, bins=bins, cumulative=True, histtype="step", label="blue", color="b")
        ax1[0].set_title("RGB Cumulative Curve")

        ax1[1].hist(img_r, histtype="step", label="red", color="r")
        ax1[1].hist(img_g, histtype="step", label="green", color="g")
        ax1[1].hist(img_b, histtype="step", label="blue", color="b")
        ax1[1].set_title("RGB Distribution Curve")

        output_folder = os.getcwd()
        distribution_path = os.path.join(output_folder, 'RGB_Distribution&Cumulative_Curve.png')
        plt.savefig(distribution_path)
        pixmap = QPixmap("RGB_Distribution&Cumulative_Curve.png")
        resized_pixmap = pixmap.scaled(self.ui.equalize_dist.size())
        self.ui.rgb_dist.setPixmap(resized_pixmap)

        plt.close()

    def draw_histogram(self, img,type):
        img = np.asarray(img)
        # put pixels in a 1D array by flattening out img array
        flat = img.flatten()
        plt.figure()
        plt.hist(flat, 50)
        plt.title('Histogram')
        output_folder = os.getcwd()  # Current working directory
        if not type:
            histogram_path = os.path.join(output_folder, 'equalize_histogram.png')
            plt.savefig(histogram_path)
            pixmap = QPixmap("equalize_histogram.png")
            resized_pixmap = pixmap.scaled(self.ui.equalize_histo.size())
            self.ui.equalize_histo.setPixmap(resized_pixmap)
        else:
            histogram_path = os.path.join(output_folder, 'Normalize_histogram.png')
            plt.savefig(histogram_path)
            pixmap = QPixmap("Normalize_histogram.png")
            resized_pixmap = pixmap.scaled(self.ui.equalize_histo.size())
            self.ui.normaliz_histo_4.setPixmap(resized_pixmap)

        plt.close()

    def draw_distribution(self,img,type):
        """
        Plot the distribution curve for the given image.

        Args:
            img: Input image.
            type: Type of distribution curve (0 for equalization, 1 for normalization).
        """
        plt.figure()
        plt.hist(img.ravel(), bins=256, cumulative=True)
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        output_folder = os.getcwd()  # Current working directory
        if not type:
            dist_path = os.path.join(output_folder, 'equalize_distribution_curve.png')
            plt.savefig(dist_path)
            pixmap = QPixmap("equalize_distribution_curve.png")
            resized_pixmap = pixmap.scaled(self.ui.equalize_dist.size())
            self.ui.equalize_dist.setPixmap(resized_pixmap)

        else:
            dist_path = os.path.join(output_folder, 'Normalize_distribution_curve.png')
            plt.savefig(dist_path)
            pixmap = QPixmap("Normalize_distribution_curve.png")
            resized_pixmap = pixmap.scaled(self.ui.equalize_dist.size())
            self.ui.normalize_dist_4.setPixmap(resized_pixmap)

        plt.close()

    def thresholding(self,image):
        """
        Apply thresholding to the image based on user-defined parameters.

        Args:
            image: Input image.
        """
        if image is None:
            print('Please enter an image')

        else:

            if (image.ndim == 3):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif (image.ndim == 2):
                gray = image

            if self.ui.combo_thersholding.currentIndex()==0:
                self.ui.slider_2.show()
                self.ui.slider_3.show()
                self.ui.slider_4.show()
                final_img = local_thresholding(gray,self.ui.slider_1.value(),self.ui.slider_2.value(),
                                               self.ui.slider_3.value(),self.ui.slider_4.value())


            else:

            # Applying the threshold on the image whether it is calculated or given by the user according to the previous condition

                self.ui.slider_2.hide()
                self.ui.slider_3.hide()
                self.ui.slider_4.hide()
                final_img=global_thresholding(gray,self.ui.slider_1.value())

            display_image(self.ui.thersolding_img,final_img.astype(np.uint8))


    ###### tab 3  ######
    def browse(self,flag):
        """
        Browse and display images based on the flag (0 for low-pass, 1 for high-pass).

        Args:
            flag: Flag indicating the type of image (0 for low-pass, 1 for high-pass).
        """
        if self.ui.tabWidget.currentIndex()==2:
            if flag:
                self.before_high_image = browse(self)
                display_image(self.ui.orig_img_hypred_2, self.before_high_image)
                self.ui.hpf_img.clear()

            else:
                self.before_low_image = browse(self)
                display_image(self.ui.orig_img_hypred_1, self.before_low_image)
                self.ui.LPF_img.clear()

            self.handle_spinbox(flag)

        if self.ui.tabwidget_3.currentIndex()==2:
            if flag:
                self.match_2 = browse(self)
                display_image(self.ui.orig_img_matching2, self.match_2)


            else:
                self.match_1 = browse(self)
                display_image(self.ui.orig_img_matching1, self.match_1)

        if self.ui.tabWidget_3.currentIndex()==0:
            if flag:
                # self.sift_img1 = browse(self)
                file_dialog = QFileDialog()
                self.sift_img1, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
                img = cv2.imread(self.sift_img1)
                display_image(self.ui.sift_img_1, img)
            else:
                # self.sift_img2 = browse(self)
                file_dialog = QFileDialog()
                self.sift_img2, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
                img = cv2.imread(self.sift_img2)

                display_image(self.ui.sift_img_2, img)
  
    def convert(self):
        """
        Convert the image based on the selected conversion type.
        """
        if self.ui.comboBox.currentIndex()==1:
            image= cv2.cvtColor(self.before_conversion, cv2.COLOR_BGR2GRAY)
            output_filename = "gray_image.png"  # Change the file extension as needed
            output_path = os.path.join(os.getcwd(), output_filename)
            cv2.imwrite(output_path, image )
            pixmap = QPixmap("gray_image.png")
            self.ui.orig_img_2.setPixmap(pixmap)
            # Resize the QLabel to fit the image
            self.ui.orig_img_2.setScaledContents(True)

        else:
            image=self.before_conversion
            display_image(self.ui.orig_img_2,image)

        self.equalization(image)
        self.normalize(image)

    def handle_spinbox(self,flag):
        """
        Handle spin box changes for low-pass and high-pass filtering.

        Args:
            flag: Flag indicating the type of image (0 for low-pass, 1 for high-pass).
        """

        if self.before_low_image is not None:
            if not flag:
                low_pass(self.before_low_image, self.ui.spin_lp_cuttoff.value())
                output = gaussian_blur(self.before_low_image, self.ui.spin_lp_cuttoff.value())
                output_filename = "low_image.png"  # Change the file extension as needed
                output_path = os.path.join(os.getcwd(), output_filename)
                cv2.imwrite(output_path, output * 255)
                pixmap = QPixmap("low_image.png")
                resized_pixmap = pixmap.scaled(self.ui.LPF_img.size())
                self.ui.LPF_img.setPixmap(resized_pixmap)

        else:
            print("Original image is not available. Please browse and select an image.")

        if self.before_high_image is not None:
            if flag:
                output = high_pass(self.before_high_image, self.ui.spin_hp_cuttoff.value())
                output_filename = "high_image.png"  # Change the file extension as needed
                output_path = os.path.join(os.getcwd(), output_filename)
                cv2.imwrite(output_path, output*255)
                pixmap = QPixmap("high_image.png")
                resized_pixmap = pixmap.scaled(self.ui.hpf_img.size())
                self.ui.hpf_img.setPixmap(resized_pixmap)
        else:
            print("Original image is not available. Please browse and select an image.")

    def hybrid_image(self):
        """
        Generate a hybrid image from the selected low-pass and high-pass images.
        """
        if self.before_high_image is not None and self.before_low_image is not None:
            low = low_pass(self.before_low_image, self.ui.spin_lp_cuttoff.value())
            high = high_pass(self.before_high_image, self.ui.spin_hp_cuttoff.value())

            low_resized = cv2.resize(low, (high.shape[1], high.shape[0]))
            result = low_resized + high

            output_filename = "result.png"  # Change the file extension as needed
            output_path = os.path.join(os.getcwd(), output_filename)
            cv2.imwrite(output_path, result * 255)
            pixmap = QPixmap("result.png")
            resized_pixmap = pixmap.scaled(self.ui.hybers_img.size())
            self.ui.hybers_img.setPixmap(resized_pixmap)

            print("Creating hybrid image...")
        else:
            print("Not enough images")



######################################Task 2#################################



###canny Tab ###
    def canny_detection(self, img):
        if img is None:
            print("please enter image")
        else:
            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            canny_image = findEdges(image_gray, self.ui.spinBox_canny_kernel.value(), self.ui.spinBox_sigma.value())
            display_image(self.ui.output_img_canny,canny_image.astype(np.uint8))
###line Tab ###

    def hough_line(self, img):
        if img is None:
            print("please enter image")
        else:
            color = self.ui.comboBox_color_line.currentText()
            num_lines = self.ui.spinBox__line.value()
            T_low = self.ui.Lth_Slider.value()
            T_high = self.ui.Hth_Slider.value()
            neighborhood_size = self.ui.neigh_Slider.value()
            print("################")
            print("T_low:", T_low)
            print("color:", color)
            print("T_high:", T_high)
            print("neighborhood_size:", neighborhood_size)
            print("num_lines:", num_lines)
            print("img:", img)




            print("################")

            line_image = hough_lines(T_low, T_high, neighborhood_size, img, num_lines)
            display_image(self.ui.out_img__line,line_image.astype(np.uint8))

## circle tab ##
    def apply_circle(self  ):
        r_min = self.ui.minR_Slider.value()
        r_max = self.ui.maxR_Slider.value()
        delta_r = self.ui.deltaR_Slider.value()
        num_thetas = self.ui.num_thetas_Slider.value()
        bin_threshold = self.ui.bin_Slider.value()/10
        min_edge_threshold = self.ui.min_edge_threshold_Slider.value()
        max_edge_threshold = self.ui.max_edge_threshold_Slider.value()
        filter_size = self.ui.filterSize_Slider.value()
        circle_img, circles = find_hough_circles(self.original_image, r_min, r_max, delta_r, num_thetas, bin_threshold, self.ui.circle_progressBar , filter_size, min_edge_threshold, max_edge_threshold, False)
        display_image(self.ui.out_img__circle, circle_img)


## ellipse tab ##
    def apply_ellipse(self):
        a_min = int(self.ui.doubleSpinBox_a_min.value())  #5
        b_min = int(self.ui.doubleSpinBox_b_min.value())
        a_max = int(self.ui.doubleSpinBox_a_max.value())   #5
        b_max = int(self.ui.doubleSpinBox_b_max.value())
        delta_a = int(self.ui.doubleSpinBox_delta_a.value()) #2
        delta_b = int(self.ui.doubleSpinBox_delta_b.value())
        num_thetas = int(self.ui.doubleSpinBox_theta_num.value()) 
        bin_threshold = self.ui.doubleSpinBox_bin_th.value()
        filter_size = self.ui.ellipse_filter_size.value()
        min_edge = self.ui.min_edge_th_ellipse.value()
        max_edge = self.ui.max_edge_th_ellipse.value()

        # out , _ = find_hough_ellipses(self.original_image,  a_min , a_max, delta_a , b_min , b_max, delta_b, num_thetas , bin_threshold , filter_size , min_edge , max_edge, self.ui.progressBar_ellipse,  post_process=True)
        print(a_min , a_max , delta_a , b_min , b_max , delta_b , num_thetas , filter_size , min_edge , max_edge , bin_threshold)
        out , _ = find_hough_ellipses(self.original_image , a_min , a_max , delta_a , b_min , b_max , delta_b , num_thetas , filter_size , min_edge , max_edge , bin_threshold , self.ui.progressBar_ellipse , True)
        # out , _ = find_hough_ellipses(self.original_image,  a_min=5 , a_max=100, delta_a=2, 
        #                                                 b_min=5 , b_max=100, delta_b=2, num_thetas=20 , 
        #                                                 bin_threshold=0.7, post_process=False)
        display_image(self.ui.output_img_ellipse , out)

####snake rab####
    def active_contour(self,img):
        if img is None:
            print("please enter image")
        else:
            snake = sn.Snake(img,alpha=self.ui.alpha_Slider.value(),
                             beta=self.ui.beta_Slider.value(),
                             gamma=self.ui.gamma_Slider.value()
            )

            for i in range (self.ui.spinBox_snake.value()):
                img = snake.visuaize_Image()
                display_image(self.ui.out_img__snake,img.astype(np.uint8))


                snake_changed = snake.step()
                x = []
                y = []
                for i in range(len(snake.points)):
                    x.append(snake.points[i][0])
                    y.append(snake.points[i][1])
                area = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
                area = np.abs(area)
                perimeter = snake.get_length()
                self.ui.label_12.setText("{}".format(area / 10000))
                self.ui.label_20.setText("{}".format(perimeter / 100))
                QtWidgets.QApplication.processEvents()





########################################Task 3###########################################

#### harris ####

    def harris_match(self):
        start = time.time()
        self.harris_output = apply_harris(self.harris_input, float(self.ui.spinBox_sensitivity.text()),
                                          float(self.ui.spinBox_harris_threshold.text()))
        # self.harris_output = cv2.cvtColor(self.harris_output, cv2.COLOR_BGR2RGB)
        end = time.time()
        display_image(self.ui.output_img_harris, self.harris_output)
        self.ui.lcdNumber_harris.display(end-start)

#### match ####
    def match(self, image1, image2):
        if image1 is None or image2 is None:
            print("not enough images")
        else:
            self.ui.out_img__matching.clear()
            if self.ui.radioButton_SSD.isChecked():
                self.ui.process.setText('Processing...........')
                QCoreApplication.processEvents()
                computation_time, matched_image = handle_matching(image1, image2, self.ui.spinBox_numMatch.value(), 0)
                display_image(self.ui.out_img__matching, matched_image)
                self.ui.lcdNumber_maching.display(computation_time)
                self.ui.process.setText('Done processing')
                QCoreApplication.processEvents()
            else:
                self.ui.process.setText('Processing...........')
                QCoreApplication.processEvents()
                computation_time, matched_image = handle_matching(image1, image2, self.ui.spinBox_numMatch.value(), 1)
                display_image(self.ui.out_img__matching, matched_image)
                self.ui.lcdNumber_maching.display(computation_time)
                self.ui.process.setText('Done processing')
                QCoreApplication.processEvents()
#### Sift ######
    def apply_sift(self):
        start = time.time()
        img1 = cv.imread(self.sift_img1, 0)           # queryImage
        img2 = cv.imread(self.sift_img2, 0)  # trainImage

        sift1 = MY_SIFT(img1)
        sift2 = MY_SIFT(img2)
        
        keypoints1, descriptors1 = sift1.computeKeypointsAndDescriptors()
        keypoints2, descriptors2 = sift2.computeKeypointsAndDescriptors()

        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Draw matches
        matched_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        display_image(self.ui.out_img__sift , matched_img)
       
        
        
        end = time.time()
        self.ui.lcdNumber_sift.display(end-start)

######################################Task 4#################################

    def optimal_thresholding(self, image):
        if image is None:
            print("not enough images")
        else:
            image = np.copy(self.original_image)
            if self.ui.radioButton_local_opt.isChecked():
                Optimized_Thresholding(image, 'Local')
                thresholding_image = cv2.imread('./images/optimal_out2.png')
                # Display the image in the widget
                display_image(self.ui.output_img_optimal, thresholding_image)
            else:
                Optimized_Thresholding(image, 'Global')
                thresholding_image = cv2.imread('./images/optimal_out1.png')
                # Display the image in the widget
                display_image(self.ui.output_img_optimal, thresholding_image)

    def otsu_thresholding(self, image):
        if image is None:
            print("not enough images")
        else:
            image = np.copy(self.original_image)
            if self.ui.radioButton_local_otsu.isChecked():
                local_thresholding(image)
                thresholding_image = cv2.imread("images/local.png")
                display_image(self.ui.out_img__otsu, thresholding_image)
            else:
                global_thresholding(image)
                thresholding_image = cv2.imread("images/global.png")
                # Display the image in the widget
                display_image(self.ui.out_img__otsu, thresholding_image)

    def manual_thresholding(self,image):
        """
        Apply thresholding to the image based on user-defined parameters.

        Args:
            image: Input image.
        """
        if image is None:
            print('Please enter an image')

        else:
            image = np.copy(self.original_image)

            if (image.ndim == 3):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif (image.ndim == 2):
                gray = image

            if self.ui.comboBox_2.currentIndex()==0:
                self.ui.slider_5.show()
                self.ui.slider_7.show()
                self.ui.slider_8.show()
                final_img = th.local_thresholding(gray,self.ui.slider_6.value(),self.ui.slider_5.value(),
                                               self.ui.slider_7.value(),self.ui.slider_8.value())
               

                

            else:

            # Applying the threshold on the image whether it is calculated or given by the user according to the previous condition

                self.ui.slider_5.hide()
                self.ui.slider_8.hide()
                self.ui.slider_7.hide()
                final_img=th.global_thresholding(gray,self.ui.slider_6.value())
            display_image(self.ui.out_img__manual,final_img.astype(np.uint8))
    
    def spectral_thresholding(self, img):
        if img is None:
            print("please enter image")
        else:
            
            image = np.copy(self.original_image)
            if self.ui.radioButton_global_spec.isChecked():
                thresholding_image=spectral_threshold(image)

                # Display the image in the widget
                display_image(self.ui.out_img_spec,thresholding_image)

            else :
                thresholding_image= local_spectral_thresholding(image, 0, 0, 0, 0,2)
                display_image(self.ui.out_img_spec, thresholding_image)
            print("################")
    
    
    
    def mean_shift(self,image):
        if image is None:
            print("not enough images")
        else:
            meanshift_obj = meanshift(image,self.ui.horizontalSlider_iter_mean.value())
            output_image = meanshift_obj.performMeanShift(image)
            display_image(self.ui.out_img__shift,output_image.astype(np.uint8))

    def agglomrative(self,image):
        if image is None:
            print("not enough images")
        else:
            output_image=apply_agglomerative_clustering( self.ui.clusteringnum.value(), self.ui.initial_clustering.value(), image)
            display_image(self.ui.out_img_agg,output_image)

    def apply_kmeans(self):
        '''
        write your pratmers here
        this is the premters in the ui
        '''

        num_of_iterations = self.ui.slider_iter_k.value()
        cluster = self.ui.horizontalSlider_clus_k.value()

        output_image = kmeans(self.original_image,  num_of_iterations ,cluster)
        display_image(self.ui.output_img_k, output_image)

    def apply_regoin(self , event):
        
        
        label_width = self.ui.out_img__reg.width()
        label_height = self.ui.out_img__reg.height()

        # Get the width and height of the displayed image
        image_height , image_width = self.original_image.shape[:2]
        

        # image_height = self.original_image.height()
        
        x_scale = image_width / label_width
        y_scale = image_height / label_height

        # Map the coordinates of the mouse event    to the corresponding coordinates on the image

        if event.button() == Qt.LeftButton:
    
            x = int(event.x() * x_scale )
            y = int(event.y() * y_scale)
            print(x , y)
            threshold_reg =int(self.ui.spinBox.value())
            output_image = region_growing(self.original_image,(y , x) , threshold_reg)
            display_image(self.ui.out_img__reg , output_image)

    def luv_map(self, image):
        output_image = rgb_to_luv_man(image)
        display_image(self.ui.out_img__luv , output_image)

    ######################################Task 5#################################

    def apply_detect(self):
        scale_factor = self.ui.spinBox_scale.value()
        min_neighbours = self.ui.spinBox_neighbours.value()

        face_cascade = load_face_cascade()

        resized_image = cv2.resize(self.original_image, (500, 500))
        image = detect_and_draw_faces(resized_image, face_cascade)
        output = convert_to_rgb(image)
        display_image(self.ui.output_img_detect , output)


    # def face_detection(self, img):
    #     error_path = 'our_faces/test_images/404.jpg'
    #     print("kkkkkk")
    #
    #     model = train_svm_with_pca()
    #
    #
    #     if img is not None:
    #         path = 'our_faces/test_images/'
    #         path_1 = path + img.name
    #         print (path_1)
    #         image_1 = cv2.imread(path_1)
    #         height, width = image_1.shape[1], image_1.shape[0]
    #
    #         image, prediction, found = predict_with_svm(model, path_1)
    #         if found >= 0.5:
    #             out_put = image
    #         else:
    #             out_put = cv2.imread(error_path, 0)
    #         prediction = int(prediction) - 1
    #         # st.markdown(names[prediction])
    #         display_image(self.ui.output_img_detect , out_put)

    def Recognize(self):
        print('Reco')
        self.outputImages = 0
        # self.faces=FaceDetection(self.images[index])
        # if len(self.faces) > 0:
        t1 = time.time()
        img = np.copy(self.original_image)
        pca(img)
        print('Reco')

        # Call your function here
        t2 = time.time()
        if os.path.isfile('our_faces/FaceRecognized.png'):
            output_pixmap = cv2.imread('our_faces/FaceRecognized.png')
            display_image(self.ui.out_img__recogn, output_pixmap)

        # output = cv2.imread("our_faces/data/FaceRecognized.png")
        self.ui.lcdNumber_recogn.display(t2-t1)

        # self.label_2.setText("Computation Time: "+str(round((t2-t1),3))+"Sec")
    def roc(self):
        start = time.time()
        classnum = self.ui.spinBox_class.value()
        y, prob_vector, _, acc = create_model()
        _, _, thresh = calc_fpr_tpr_thresh(classnum)
        CM = calc_CM(prob_vector[:, classnum-1], y[:, classnum-1], thresh[classnum])
        draw_CM(CM)
        end = time.time()
        computation_time=end-start
        self.ui.lcdNumber_evaul.display(computation_time)
        pixmap = QPixmap("our_faces/ROC2.png")
        resized_pixmap = pixmap.scaled(self.ui.image_input_roc.size())
        self.ui.image_input_roc.setPixmap(resized_pixmap)
        pixmap = QPixmap("our_faces/CM.png")
        resized_pixmap = pixmap.scaled(self.ui.image_out_roc.size())
        self.ui.image_out_roc.setPixmap(resized_pixmap)
        self.ui.label_accuracy.setText("{}".format(acc))




def main():
    app = QApplication(sys.argv)
    window = MyWindow() 
   
   
    window.showMaximized()
    window.show()
    sys.exit(app.exec_())
    
    

if __name__ == '__main__':
    main()