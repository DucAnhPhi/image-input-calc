import cv2 as cv
import numpy as np


class PreProcessing:
    def convert_gray(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def gaussian_blur(self, img):
        return cv.GaussianBlur(img, (7, 7), 0)

    def median_blur(self, img):
        return cv.medianBlur(img, 3)

    def erode(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv.erode(img, kernel, iterations=1)

    def dilate(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv.dilate(img, kernel, iterations=1)

    def morph_open(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)

    def morph_close(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)

    def morph_gradient(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

    def binarize(self, img):
        # adaptive gaussian thresholding
        binarized = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY, 11, 2)
        return binarized

    def preprocess(self, img):
        preprocessed = self.convert_gray(img)
        preprocessed = self.morph_open(preprocessed)
        preprocessed = self.gaussian_blur(preprocessed)
        preprocessed = self.binarize(preprocessed)
        # preprocessed = self.erode(preprocessed)
        preprocessed = self.morph_close(preprocessed)
        return preprocessed
