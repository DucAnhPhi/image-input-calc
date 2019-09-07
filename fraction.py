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

    def get_contour(self):
        # a contour is a np.array with shape (#points, 1, 2)
        # in which each entry represents (x,y) coordinates of boundary points
        mask = np.ones(self.frameShape[:2], dtype="uint8") * 255
        cv.rectangle(mask, (self.x1, self.y1),
                     (self.x2, self.y2), (0, 0, 0), 1)
        cnts, _ = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        return cnts[1]
