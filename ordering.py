import cv2 as cv
import numpy as np

from segmentation import Segmentation
from drawing import Draw
from contour import Contour
from fraction import Fraction


class LineOrdering:
    def get_max_vec(self, vectors):
        maxDist = 0
        maxVec = None
        for vec in vectors:
            tempDist = np.linalg.norm(vec)
            if tempDist > maxDist:
                maxDist = tempDist
                maxVec = vec
        return maxVec

    def normalize_vec(self, v):
        normalized = v
        norm = np.linalg.norm(v)
        if norm > 0:
            normalized = v / norm
        return normalized

    def redirect_vectors(self, vectors):
        # get vector with max distance
        maxVec = self.get_max_vec(vectors)
        redirectedVectors = []
        for vec in vectors:
            normalized = self.normalize_vec(vec)
            if vec.dot(maxVec) > 0:
                redirectedVectors.append(normalized)
            else:
                redirectedVectors.append(normalized * (-1))
        return redirectedVectors

    def get_hor_vec(self, contourList):
        # Concept:
        # We want to determine the direction in which the line is written.
        # The Problem is that our Input is likely to not be perfect.
        # Very likely we will have very many random dots and some large contours which do not belong into the line.
        # For determining the horizontal Vector (horVec) we will use a list of all possible vectors pointing from one contour to another.
        # Because a significant percentage of these will be between in-line variables, we can determine the horVec by calculating the mean.
        #
        # ToDo: There could be a problem if the largestVector is orthogonal to the line.

        # get distance vectors
        distVectors = []
        for i in range(len(contourList)):
            currCnt = contourList[i]
            for j in range(i+1, len(contourList)):
                nextCnt = contourList[j]
                distVec = nextCnt.center - currCnt.center
                distVectors.append(distVec)

        # point all vectors to dominant direction
        redirected = self.redirect_vectors(distVectors)

        horVec = np.median(redirected, axis=0)
        horVec = horVec * 100
        # (x,y) shape
        horVec = self.normalize_vec(horVec)
        return horVec

    def get_orth_vec(self, contourList):
        horVec = self.get_hor_vec(contourList)
        # We want to find a vector (xO,yO) which is orthogonal to (xH,yH).
        # This is equivalent (assuming we have non-vanishing vectors) to:
        #   xHxO+yOyH=0
        # The solution for this is:
        #   xO = - yH * a
        #   yO = xH * a
        #   with a not being 0.
        # 'a' can be 1 or -1
        orthVec = np.array([-horVec[1], horVec[0]])
        return orthVec
