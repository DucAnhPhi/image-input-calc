import cv2
import numpy as np

class PreProcessing:
    def convert_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def blur_and_binarize(self, img):
        blur = cv2.GaussianBlur(img, (9,9), 0)
        # adaptive gaussian thresholding
        binarized = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        return binarized

    def preprocess(self, img):
        preprocessed = self.convert_gray(img)
        preprocessed = self.blur_and_binarize(preprocessed)
        return preprocessed