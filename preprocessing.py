import cv2
import numpy as np

class PreProcessing:
    def convert_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def gaussian_blur(self, img):
        return cv2.GaussianBlur(img,(7,7),0)
    
    def median_blur(self, img):
        return cv2.medianBlur(img, 3)
    
    def morph_open(self, img):
        kernel = np.ones((3,3),np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations= 1)

    def morph_close(self, img):
        kernel = np.ones((3,3),np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 1)

    def morph_gradient(self, img):
        kernel = np.ones((3,3),np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    def binarize(self, img):
        # adaptive gaussian thresholding
        binarized = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        return binarized

    def preprocess(self, img):
        preprocessed = self.convert_gray(img)
        #preprocessed = self.morph_transformations(preprocessed)
        preprocessed = self.gaussian_blur(preprocessed)
        #preprocessed = self.morph_open(preprocessed)
        #preprocessed = self.morph_gradient(preprocessed)
        preprocessed = self.binarize(preprocessed)
        #preprocessed = self.morph_close(preprocessed)
        #preprocessed = self.median_blur(preprocessed)
        return preprocessed