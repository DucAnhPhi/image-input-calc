import cv2 as cv
import numpy as np


class Fraction:
    def __init__(self, bar, contourList, frameShape):
        self.frameShape = frameShape  # shape of original frame
        self.bar = bar  # contour object
        self.nominator = []  # list of contour objects
        self.denominator = []  # list of contour objects

        # define acceptance area
        minX = bar.x1
        maxX = bar.x2
        minY = max(bar.y1-(bar.radius), 0)
        maxY = bar.y2+(bar.radius)
        center = (maxY-minY) // 2 + minY

        # boundary coordinates of bounding box around fraction
        self.x1 = minX
        self.x2 = maxX
        self.y1 = max(frameShape)
        self.y2 = 0

        # look for contours inside acceptance area
        for nb in contourList:
            if nb.contourId == bar.contourId:
                continue
            if nb.is_inside_area(minX, maxX, minY, center):
                self.nominator.append(nb)
                # update boundary coordinates
                self.x1 = min(self.x1, nb.x1)
                self.x2 = max(self.x2, nb.x2)
                self.y1 = min(self.y1, nb.y1)
                continue
            if nb.is_inside_area(minX, maxX, center, maxY):
                self.denominator.append(nb)
                # update boundary coordinates
                self.x1 = min(self.x1, nb.x1)
                self.x2 = max(self.x2, nb.x2)
                self.y2 = max(self.y2, nb.y2)

        # sort contours horizontally for now
        # TODO: replace by more sophisticated ordering
        self.nominator.sort(key=lambda cnt: cnt.x1)
        self.denominator.sort(key=lambda cnt: cnt.x1)

    def get_contour(self):
        # a contour is a np.array with shape (#points, 1, 2)
        # in which each entry represents (x,y) coordinates of boundary points
        top = []
        bottom = []
        left = []
        right = []
        for i in range(self.x2-self.x1-1):
            if i == 0:
                continue
            top.append([self.x1 + i, self.y1])
            bottom.append([self.x2 - i, self.y2])
        for i in range(self.y2-self.y1-1):
            if i == 0:
                continue
            right.append([self.x2, self.y1 + i])
            left.append([self.x1, self.y2 - i])
        contour = [*top, *right, *bottom, *left]
        contour = np.array(contour).astype(np.int32)
        return contour
