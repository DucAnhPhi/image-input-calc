import cv2 as cv
import numpy as np

from segmentation import Segmentation
from drawing import Draw
from contour import Contour
from fraction import Fraction


class LineOrdering:
    def __init__(self, contourList):
        self.contourList = contourList
        # self.horVec = self.get_hor_vec()
        self.horVec = (10, 0)
        self.orthVec = self.get_orth_vec(self.horVec)

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

    def get_hor_vec(self):
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
        for i in range(len(self.contourList)):
            currCnt = self.contourList[i]
            for j in range(i+1, len(self.contourList)):
                nextCnt = self.contourList[j]
                distVec = nextCnt.center - currCnt.center
                distVectors.append(distVec)

        # point all vectors to dominant direction
        redirected = self.redirect_vectors(distVectors)

        horVec = np.median(redirected, axis=0)
        horVec = horVec * 100
        # (x,y) shape
        horVec = self.normalize_vec(horVec)
        return horVec

    def get_orth_vec(self, horVec):
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

    def get_avg_y_dev(self):
        avgYDev = 0
        n = len(self.contourList)
        for i in range(n):
            if i == 0:
                continue
            currentY = self.contourList[i].center[1]
            preY = self.contourList[i-1].center[1]
            avgYDev += abs(currentY-preY)
        return avgYDev / n

    def separate_into_lines(self, avgYDev):
        lines = []
        tmpLine = []
        cnts = self.contourList
        n = len(cnts)
        for i in range(n):
            current = cnts[i]
            if i == 0:
                tmpLine.append(current)
                continue
            pre = cnts[i-1]
            currentY = current.center[1]
            preY = pre.center[1]
            yDev = abs(currentY-preY)
            if yDev <= avgYDev:
                tmpLine.append(current)
            else:
                print(i)
                lines.append(tmpLine)
                tmpLine = [current]
            if i == n-1:
                lines.append(tmpLine)
        return lines

    def get_lines(self, frame):
        cnts = self.contourList
        # sort contours by Y coordinate of their centroids
        cnts.sort(key=lambda cnt: cnt.center[1])

        # compute average Y deviation from neighbouring contour
        avgYDev = self.get_avg_y_dev()

        # separate ordered list to lines where y deviation is above average
        lines = self.separate_into_lines(avgYDev)

        # order contours in a line by x coordinate of their centroids
        for l in range(len(lines)):
            line = lines[l]
            line.sort(key=lambda cnt: cnt.center[0])
            for i in range(len(line)):
                cnt = line[i]
                cv.putText(frame, str(l) + str(i), (cnt.center[0], cnt.center[1]),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    def get_lines_with_hor_vec(self, frame):
        tempContours = self.contourList.copy()

        def get_orth_dist(vec):
            return np.linalg.norm(np.multiply(vec, self.orthVec))

        tempContours.sort(key=lambda cnt: get_orth_dist(cnt.center))

        avgOrthDev = 0
        for i in range(len(tempContours)):
            if i == 0:
                continue
            currDist = get_orth_dist(tempContours[i].center)
            preDist = get_orth_dist(tempContours[i-1].center)
            avgOrthDev += abs(currDist-preDist)
        avgOrthDev = avgOrthDev / len(tempContours)

        lines = []
        tmpLine = []
        for i in range(len(tempContours)):
            curr = tempContours[i]
            if i == 0:
                tmpLine.append(curr)
                continue
            pre = tempContours[i-1]
            currDist = get_orth_dist(curr.center)
            preDist = get_orth_dist(pre.center)
            orthDev = abs(currDist-preDist)
            if orthDev <= avgOrthDev:
                tmpLine.append(curr)
            else:
                if len(tmpLine) > 2:
                    lines.append(tmpLine)
                tmpLine = []

        print(self.horVec, self.orthVec)

        def get_hor_dist(vec):
            return np.linalg.norm(np.multiply(vec, self.horVec))

        for l in range(len(lines)):
            line = lines[l]
            line.sort(key=lambda cnt: get_hor_dist(cnt.center[0]))
            for i in range(len(line)):
                cnt = line[i]
                cv.putText(frame, str(l) + str(i), (cnt.center[0], cnt.center[1]),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
