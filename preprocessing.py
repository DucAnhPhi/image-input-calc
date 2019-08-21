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

    def mediumbinarize(self, img):
        # We add a specialized threshold to binarize
        boundary = (np.amin(img) + np.amax(img)) / 2
        _, medianbinarized = cv.threshold(
            img, boundary, 255, cv.THRESH_BINARY)
        return medianbinarized

    def preprocess(self, img):
        preprocessed = self.convert_gray(img)
        preprocessed = self.gaussian_blur(preprocessed)
        preprocessed = self.morph_open(preprocessed)
        preprocessed = self.binarize(preprocessed)
        preprocessed = self.erode(preprocessed)
        return preprocessed

    def mask_preprocess(self, img):
        preprocessed = self.convert_gray(img)
        preprocessed = self.gaussian_blur(preprocessed)
        preprocessed = self.morph_open(preprocessed)
        #preprocessed = self.morph_gradient(preprocessed)
        preprocessed = self.binarize(preprocessed)
        preprocessed = self.morph_close(preprocessed)
        return preprocessed

    def BorderRemovalMask(self, InIm, kernelsize=20, iterating=10):  # 5,5
        # We try to remove any and all contours which connect to the border, because letters tend not to connect and most others do.
        # Create a black border. This is important for eroding and floodfilling later
        RetIm = cv.copyMakeBorder(
            InIm, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 0))

        # We erode to ensure that we don't miss isolated points
        # Please note that we erode the image so strongly, that it is no longer usefull.
        # Instead we use this to create a fairly good mask
        kernel = np.ones((kernelsize, kernelsize))
        RetIm = cv.erode(RetIm, kernel, iterating)

        # Other filters that I found might be usefull. They didn't make the cut though
        # cv.morphologyEx(PLIm, cv.MORPH_OPEN, kernel, iterations=1)
        # PLIm = cv.GaussianBlur(PLIm, (9, 9), 0)

        # IMPORTANT: We use mediumbinarize not binarize here, because we have a high contrast image.
        # Using binarize would give contours to the outside contours, which we don't want.
        RetIm = self.mediumbinarize(RetIm)

        # creating a mask for flood fill
        h = RetIm.shape[0]
        w = RetIm.shape[1]

        mask = np.zeros((h + 2, w + 2), np.uint8)

        # we flood fill the border, removing any and all contours near it. Ideally only unique features in the centre of the image remains
        cv.floodFill(RetIm, mask, (1, 1), 255)

        # We remove the border to ensure that our image has the same dimensions as it originally had
        RetIm = RetIm[10:h - 10, 10:w - 10]

        return RetIm

    def background_contour_removal(self, frame):
        # better preprocessing for clearer image
        framePreprocessed = PreProcessing().preprocess(frame)
        # better preprocessing for mask generation
        maskPreprocessed = PreProcessing().mask_preprocess(frame)
        # mask generation
        borderRemovalMask = PreProcessing().BorderRemovalMask(maskPreprocessed)
        preprocessed = np.where(borderRemovalMask == 0, framePreprocessed, 255)
        return preprocessed
