import cv2 as cv
import numpy as np
import string
import random


class Contour:
    def __init__(self, contour, imgShape, fraction=None):
        self.contour = contour

        # generate random id
        chars = string.ascii_lowercase + string.digits
        self.contourId = ''.join(random.choices(chars, k=8))

        self.fraction = fraction  # fraction object in fraction.py
        self.imgShape = imgShape  # (height, width)

        (x, y), radius = cv.minEnclosingCircle(contour)
        self.center = np.array([int(x), int(y)])
        self.radius = int(radius)

        boundingRect = cv.boundingRect(contour)
        self.width = int(boundingRect[2])
        self.height = int(boundingRect[3])

        self.x1 = int(boundingRect[0])
        self.x2 = self.x1 + self.width
        self.y1 = int(boundingRect[1])
        self.y2 = self.y1 + self.height

    def is_inside_area(self, minX, maxX, minY, maxY):
        def is_inside_x_range(x):
            return x >= minX and x <= maxX

        def is_inside_y_range(y):
            return y >= minY and y <= maxY

        inside_x_range = is_inside_x_range(
            self.x1) or is_inside_x_range(self.x2)
        inside_y_range = is_inside_y_range(
            self.y1) or is_inside_y_range(self.y2)

        return inside_x_range and inside_y_range

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
        return above and below

    def resize_keep_ratio(self, img, size=75, interpolation=cv.INTER_AREA):
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

    def get_subimage(self):
        blankImg = np.zeros(
            shape=self.imgShape, dtype=np.uint8)
        cv.drawContours(
            blankImg, [self.contour], -1, (255, 255, 255), 1)
        cv.fillPoly(blankImg, pts=[self.contour], color=(255, 255, 255))
        subImg = blankImg[self.y1:self.y2, self.x1:self.x2]
        subImg = self.resize_keep_ratio(subImg)
        return subImg
