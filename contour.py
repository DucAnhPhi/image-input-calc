import cv2 as cv
import numpy as np
import string
import random
from preprocessing import PreProcessing
from enums import Position
from skimage import morphology, filters
import math


class Contour:
    def __init__(self,
                 contour,
                 imgShape,
                 fraction=None, frameBinary=None):
        self.contour = contour
        self.frameBinary = frameBinary

        self.mathSign = None
        self.position = Position.BASIS

        self.remove = False

        # generate random id
        chars = string.ascii_lowercase + string.digits
        self.contourId = ''.join(random.choices(chars, k=8))

        self.fraction = fraction  # fraction object in fraction.py
        self.holes = []  # list of contours (not contour objects!)
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

        minAreaRect = cv.minAreaRect(contour)
        self.trueWidth = max(minAreaRect[1])
        self.trueHeight = min(minAreaRect[1])

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

    def is_bar(self, lineThickness):
        isBar = self.trueWidth > self.trueHeight * 2
        isBar = isBar and len(self.holes) == 0
        area = cv.contourArea(self.contour)
        hull = cv.convexHull(self.contour)
        hullArea = cv.contourArea(hull)
        solidity = float(area) / hullArea
        extent = float(area) / (self.trueHeight * self.trueWidth)
        isBar = isBar and extent > 0.7 and solidity > 0.7
        isBar = isBar or self.trueWidth > self.trueHeight * 4
        return isBar

    def is_point(self, lineThickness):
        _, radius = cv.minEnclosingCircle(self.contour)
        radius = int(radius)
        contourArea = cv.contourArea(self.contour)
        minCircleArea = math.pi * radius**2
        isRound = (float(contourArea) / minCircleArea) > 0.8
        isSmall = self.trueWidth <= lineThickness * \
            3 and self.trueHeight <= lineThickness * 3
        return isRound or isSmall

    def is_vertical_bar(self):
        if self.width < self.height:
            return True

    def set_math_sign_type(self, mathSign):
        self.mathSign = mathSign

    def set_position_type(self, position):
        self.position = position

    def add_hole(self, cnt):
        self.holes.append(cnt)

    def mark_for_removal(self):
        self.remove = True

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
        mask = blankImg[self.y1:self.y2, self.x1:self.x2]
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        image = self.frameBinary[self.y1:self.y2, self.x1:self.x2]
        image = cv.bitwise_and(image, image, mask=mask)

        return image

    def skeletonize(self, img):
        binary = np.where(img == 0, 0, 1)
        skel = morphology.skeletonize(binary)
        skel = np.where(skel == False, 0, 255)
        skel = skel.astype('uint8')
        return skel

    def get_thickness(self):
        image = self.get_image()

        # sum up the image to get the area
        area = np.sum(image)

        if area == 0:
            return 0

        # skeletonize the image and sum up that image to get the total length
        skel = self.skeletonize(image)
        length = np.sum(skel)

        # divide the area by the length
        thickness = area // length

        return thickness

    def get_subimage_for_classifier(self):
        subImg = self.get_image()
        subImg = self.resize_keep_ratio(subImg)
        subImg = PreProcessing().binarize(subImg)
        subImg = np.asarray(subImg).reshape((1, 32, 32))
        return subImg
