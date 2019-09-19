import cv2 as cv
import numpy as np
import string
import random
from preprocessing import PreProcessing


class Contour:
    def __init__(self,
                 contour,
                 imgShape,
                 fraction=None,
                 holes=[],
                 isFractionBar=False,
                 isMinusSign=False,
                 isEqualBar=False,
                 isEqualSign=False):
        self.contour = contour

        # Labels
        self.isFractionBar = isFractionBar
        self.isMinusSign = isMinusSign
        self.isEqualBar = isEqualBar
        self.isEqualSign = isEqualSign

        self.remove = False

        self.equalBar = None  # contour object

        # generate random id
        chars = string.ascii_lowercase + string.digits
        self.contourId = ''.join(random.choices(chars, k=8))

        self.fraction = fraction  # fraction object in fraction.py
        self.holes = holes  # list of contours (not contour objects!)
        self.imgShape = imgShape  # (height, width)

        (x, y), radius = cv.minEnclosingCircle(contour)

        self.horDist = None
        self.orthDist = None

        self.center = np.array([int(x), int(y)])
        self.x = int(x)
        self.y = int(y)
        self.radius = int(radius)

        boundingRect = cv.boundingRect(contour)
        self.width = int(boundingRect[2])
        self.height = int(boundingRect[3])

        self.x1 = int(boundingRect[0])
        self.x2 = self.x1 + self.width
        self.y1 = int(boundingRect[1])
        self.y2 = self.y1 + self.height

    def is_inside_area(self, minX, maxX, minY, maxY, considerCenter=False):
        def is_inside_x_range(x):
            return x >= minX and x <= maxX

        def is_inside_y_range(y):
            return y >= minY and y <= maxY

        is_inside = False

        if considerCenter:
            cX = self.center[0]
            cY = self.center[1]
            is_inside = is_inside_x_range(cX) and is_inside_y_range(cY)
        else:
            inside_x_range = is_inside_x_range(
                self.x1) or is_inside_x_range(self.x2)
            inside_y_range = is_inside_y_range(
                self.y1) or is_inside_y_range(self.y2)
            is_inside = inside_x_range and inside_y_range
        return is_inside

    def is_bar(self):
        if self.width > self.height:
            minAreaRect = cv.minAreaRect(self.contour)
            trueWidth = max(minAreaRect[1])
            trueHeight = min(minAreaRect[1])
            if trueWidth > trueHeight * 2:
                return True
        return False

    def is_fraction_bar(self, contourList):
        # only consider bars
        if not self.is_bar():
            return False

        # define acceptance area of possible fraction
        minX = self.x1
        maxX = self.x2
        minY = max(self.y1-(self.radius), 0)
        maxY = self.y2+(self.radius)

        # check for contours above and below bar
        above = False
        below = False
        center = ((maxY - minY) // 2) + minY
        for cnt in contourList:
            if self.contourId == cnt.contourId:
                continue
            if cnt.is_inside_area(minX, maxX, minY, center):
                above = True
                continue
            if cnt.is_inside_area(minX, maxX, center, maxY):
                below = True
        self.isFractionBar = above and below
        return self.isFractionBar

    def check_bar_type(self, contourList):
        # don't consider contours which are about to be removed
        if self.remove:
            return False
        # only consider bars
        if not self.is_bar():
            return False

        # define acceptance area of possible fraction
        minX = self.x1
        maxX = self.x2
        minY = max(self.y1-(self.radius), 0)
        maxY = self.y2+(self.radius)

        # check for contours above and below bar
        above = False
        below = False
        # remember neighbour to compose possible equal sign later
        equalBar = None
        center = ((maxY - minY) // 2) + minY
        for cnt in contourList:
            if self.contourId == cnt.contourId:
                continue
            if cnt.is_inside_area(minX, maxX, minY, center):
                above = True
                equalBar = cnt
                continue
            if cnt.is_inside_area(minX, maxX, center, maxY):
                below = True
                equalBar = cnt

        if not above and not below:
            self.isMinusSign = True
        elif above and below:
            self.isFractionBar = True
        else:
            self.isEqualBar = True
            self.equalBar = equalBar

    def is_outer_border(self):
        is_outer = False
        if self.x1 == 0 and self.y1 == 0:
            if self.height == self.imgShape[0] and self.width == self.imgShape[1]:
                self.remove = True
                is_outer = True
        return is_outer

    def check_holes(self, contourList, hierarchy, cnt, index):
        if cnt.remove:
            return
        parentIndex = hierarchy[0][index][-1]
        if parentIndex == -1:
            return
        else:
            parent = contourList[parentIndex]
            parentArea = parent.width * parent.height
            if cnt.is_inside_area(parent.x1, parent.x2, parent.y1, parent.y2):
                child_area = cnt.width * cnt.height
                if child_area > parentArea * 0.1:
                    parent.holes.append(cnt.contour)
                    cnt.remove = True

    def unwrap(self):
        # contour can contain nested contours
        # recursively unwrap these contours
        if self.fraction == None:
            return self

        def unwrap_helper(contourList):
            contours = []
            for cnt in contourList:
                if cnt.fraction == None:
                    contours.append(cnt)
                else:
                    contours.append(cnt.unwrap())
            return contours

        nominator = unwrap_helper(self.fraction.nominator)
        denominator = unwrap_helper(self.fraction.denominator)

        unwrapped = [nominator,
                     self.fraction.bar, denominator]
        return unwrapped

    def resize_keep_ratio(self, img, size=32, interpolation=cv.INTER_AREA):
        # get height and width of given image
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w:
            return cv.resize(img, (size, size), interpolation)
        # get longest edge
        dif = max(h, w)
        # calculate offsets
        xOffset = int((dif-w)/2.)
        yOffset = int((dif-h)/2.)
        # generate mask with longest edge and offsets
        if c is None:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[yOffset:yOffset+h, xOffset:xOffset+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[yOffset:yOffset+h, xOffset:xOffset+w,
                 :] = img[:h, :w, :] = img[:h, :w, :]
        # return resized mask
        return cv.resize(mask, (size, size), interpolation)

    def get_image(self):
        blankImg = np.zeros(
            shape=self.imgShape, dtype=np.uint8)
        cv.fillPoly(blankImg, pts=[self.contour, *
                                   self.holes], color=(255, 255, 255))
        image = blankImg[self.y1:self.y2, self.x1:self.x2]
        return image

    def skeletonize(self, image):
        skel = np.zeros(image.shape, np.uint8)
        size = np.size(image)

        element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        done = False

        while(not done):
            eroded = cv.erode(image, element)
            temp = cv.dilate(eroded, element)
            temp = cv.subtract(image, temp)
            skel = cv.bitwise_or(skel, temp)
            image = eroded.copy()

            zeros = size - cv.countNonZero(image)
            if zeros == size or zeros == 0:
                done = True
        return skel

    def get_thickness(self):
        image = self.get_image()
        image = PreProcessing().convert_gray(image)

        # sum up the image to get the area
        area = np.sum(image)

        # skeletonize the image and sum up that image to get the total length
        skel = self.skeletonize(image)
        length = np.sum(skel)

        # divide the area by the length
        thickness = area // length

        return thickness

    def get_subimage_for_classifier(self):
        subImg = self.get_image()
        subImg = self.resize_keep_ratio(subImg)
        subImg = PreProcessing().convert_gray(subImg)
        subImg = np.asarray(subImg).reshape((1, 32, 32))
        return subImg