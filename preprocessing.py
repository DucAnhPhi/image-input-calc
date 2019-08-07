import cv2
import numpy as np


class PreProcessing:
    def convert_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (7, 7), 0)

    def median_blur(self, img):
        return cv2.medianBlur(img, 3)

    def morph_open(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    def morph_close(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    def morph_gradient(self, img):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    def binarize(self, img):
        # adaptive gaussian thresholding
        binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                          cv2.THRESH_BINARY, 11, 2)
        return binarized

    def mediumbinarize(self, img):
        boundary = (np.amin(img) + np.amax(img)) / 2
        ret, medianbinarized = cv2.threshold(img, boundary, 255, cv2.THRESH_BINARY)
        return medianbinarized

    def redfilter(self, img):
        r = img.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        return r

    def bluefilter(self, img):
        b = img.copy()
        b[:, :, 1] = 0
        b[:, :, 2] = 0
        return b

    def greenfilter(self, img):
        g = img.copy()
        g[:, :, 0] = 0
        g[:, :, 2] = 0
        return g

    def customOpen(self,img):
        # Closing some of the Gaps
        ZS = 255 - img.copy()

        kernel = np.ones((3, 3))

        ZS = cv2.morphologyEx(ZS, cv2.MORPH_OPEN, kernel, iterations=1)

        kernel = np.ones((5, 5))
        ZS = cv2.dilate(ZS, kernel, iterations=2)

        ZS = 255 - ZS
        return ZS

    def preprocess(self, img):
        preprocessed = self.convert_gray(img)
        # preprocessed = self.morph_transformations(preprocessed)
        preprocessed = self.gaussian_blur(preprocessed)
        # preprocessed = self.morph_open(preprocessed)
        # preprocessed = self.morph_gradient(preprocessed)
        preprocessed = self.binarize(preprocessed)
        # preprocessed = self.morph_close(preprocessed)
        # preprocessed = self.median_blur(preprocessed)
        return preprocessed

    # failed attempt please ignore

    def preprocess2(self, img):
        preprocessed = self.convert_gray(img)
        # preprocessed = self.morph_transformations(preprocessed)
        preprocessed = self.gaussian_blur(preprocessed)
        preprocessed = self.morph_open(preprocessed)
        # preprocessed = self.morph_gradient(preprocessed)
        preprocessed = self.binarize(preprocessed)
        # preprocessed = self.morph_open(preprocessed)
        preprocessed = self.morph_close(preprocessed)
        # preprocessed = self.median_blur(preprocessed)


        return preprocessed

    def BorderRemovalMask(self,InIm,kernelsize=20,iterating=10): #5,5
        # We try to remove any and all contours which connect to the border, because letters tend not to connect and most others do.
        # Create a black border. This is important for eroding and floodfilling later
        RetIm = cv2.copyMakeBorder(InIm, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # We erode to ensure that we don't miss isolated points
        # Please note that we erode the image so strongly, that it is no longer usefull.
        # Instead we use this to create a fairly good mask
        kernel = np.ones((kernelsize, kernelsize))
        RetIm = cv2.erode(RetIm, kernel, iterating)

        # Other filters that I found might be usefull. They didn't make the cut though
        # cv2.morphologyEx(PLIm, cv2.MORPH_OPEN, kernel, iterations=1)
        # PLIm = cv2.GaussianBlur(PLIm, (9, 9), 0)

        # IMPORTANT: We use mediumbinarize not binarize here, because we have a high contrast image.
        # Using binarize would give contours to the outside contours, which we don't want.
        RetIm = PreProcessing().mediumbinarize(RetIm)

        # creating a mask for flood fill
        h = RetIm.shape[0]
        w = RetIm.shape[1]

        mask = np.zeros((h + 2, w + 2), np.uint8)

        # we flood fill the border, removing any and all contours near it. Ideally only unique features in the centre of the image remains
        cv2.floodFill(RetIm, mask, (1, 1), 255);

        # We remove the border to ensure that our image has the same dimensions as it originally had
        RetIm = RetIm[10:h - 10, 10:w - 10]



        return RetIm



































        #
        # Failed attempts. Simply ignore. Please don't delete, I use it for reference
        #





        ## PLIm = cv2.resize(InIm, None, fx=0.5, fy=0.5)
        '''r = img.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        r=self.convert_gray(r)
        r = self.morph_gradient(r)
        r = self.binarize(r)

        g = img.copy()
        g[:, :, 0] = 0
        g[:, :, 2] = 0
        g=self.convert_gray(g)
        g = self.morph_gradient(g)
        g = self.binarize(g)

        b = img.copy()
        b[:, :, 1] = 0
        b[:, :, 2] = 0
        b=self.convert_gray(b)
        b = self.morph_gradient(b)
        b = self.binarize(b)

        preprocessed=r*g*b

        #preprocessed = self.convert_gray(img)
        #preprocessed = self.morph_gradient(preprocessed)
        #preprocessed =self.morph_close(preprocessed)
        #preprocessed =self.morph_close(preprocessed)
        #preprocessed =self.morph_open(preprocessed)
        #preprocessed=255-preprocessed
        preprocessed = self.binarize(preprocessed)'''





