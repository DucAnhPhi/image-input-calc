import cv2 as cv
import numpy as np


class PreProcessing:
    def invert_grayscale_image(self, img):
        return (255-img)
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

    def medium_binarize(self, img):
        # We add a specialized threshold to binarize
        boundary = (np.amin(img) + np.amax(img)) / 2
        _, medianbinarized = cv.threshold(
            img, boundary, 255, cv.THRESH_BINARY)
        return medianbinarized

    def custom_binarize(self, img, boundary):
        # We add a specialized threshold to binarize
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

    def preprocess_for_mask(self, img):
        preprocessed = self.convert_gray(img)
        preprocessed = self.gaussian_blur(preprocessed)
        preprocessed = self.morph_open(preprocessed)
        #preprocessed = self.morph_gradient(preprocessed)
        preprocessed = self.binarize(preprocessed)
        preprocessed = self.morph_close(preprocessed)
        return preprocessed

    def get_background_removal_mask(self, frame, kernelsize=20, iterating=10):
        # preprocessing for mask generation
        preprocessed = self.preprocess_for_mask(frame)

        # We try to remove any and all contours which connect to the border, because letters tend not to connect and most others do.
        # Create a black border. This is important for eroding and floodfilling later
        mask = cv.copyMakeBorder(
            preprocessed, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 0))

        # We erode to ensure that we don't miss isolated points
        # Please note that we erode the image so strongly, that it is no longer usefull.
        # Instead we use this to create a fairly good mask
        kernel = np.ones((kernelsize, kernelsize))
        mask = cv.erode(mask, kernel, iterating)

        # IMPORTANT: We use mediumbinarize not binarize here, because we have a high contrast image.
        # Using binarize would give contours to the outside contours, which we don't want.
        mask = self.medium_binarize(mask)

        # creating a mask for flood fill
        h = mask.shape[0]
        w = mask.shape[1]

        floodFillMask = np.zeros((h + 2, w + 2), np.uint8)

        # we flood fill the border, removing any and all contours near it. Ideally only unique features in the centre of the image remains
        cv.floodFill(mask, floodFillMask, (1, 1), 255)

        # We remove the border to ensure that our image has the same dimensions as it originally had
        mask = mask[10:h - 10, 10:w - 10]

        return mask

    def background_contour_removal(self, frame):
        # preprocessing for clearer image
        preprocessed = self.preprocess(frame)
        # mask generation
        mask = self.get_background_removal_mask(frame)
        # apply mask on preprocessed image
        preprocessed = np.where(mask == 0, preprocessed, 255)

        # ??? remove if necessary
        #for i in range(5):
        #    preprocessed=self.erode(preprocessed)


        return preprocessed

    def preprocess3(self, img):
         preprocessed = self.convert_gray(img)
         preprocessed = self.medium_binarize(preprocessed)
         return preprocessed

    def background_contour_removal2(self, frame):
        # preprocessing for clearer image
        preprocessed = self.preprocess3(frame)
        # mask generation
        mask = self.get_background_removal_mask(frame)
        # apply mask on preprocessed image
        preprocessed = np.where(mask == 0, preprocessed, 255)

        # ??? remove if necessary
        #for i in range(5):
        #    preprocessed=self.erode(preprocessed)


        return preprocessed