import cv2 as cv
import numpy as np


class Segmentation:
    def filter_outer_border(self, img, cnt):
        x, y, width, height = cv.boundingRect(cnt)
        if x == 0 and y == 0:
            if width == img.shape[1] and height == img.shape[0]:
                return False
        return True

    def filter_nested_contour(self, img, contours, hierarchy, cnt, index, maxThresh=0.3):
        parentIndex = hierarchy[0][index][-1]
        if parentIndex == -1:
            return True
        else:
            pX, pY, pWidth, pHeight = cv.boundingRect(contours[parentIndex])
            cX, cY, cWidth, cHeight = cv.boundingRect(cnt)
            maxArea = len(img) * len(img[0]) * maxThresh
            if pWidth * pHeight < maxArea:
                if cX >= pX and cY >= pY:
                    if (cX + cWidth) <= (pX+pWidth) and (cY+cHeight) <= (pY+pHeight):
                        return False
            return True

    def filter_contours(self, img, contours, hierarchy):
        filtered = []
        for i in range(len(contours)):
            cnt = contours[i]
            valid = self.filter_outer_border(img, cnt)
            valid = valid & self.filter_nested_contour(
                img, contours, hierarchy, cnt, i)
            if valid:
                filtered.append(cnt)
        return filtered
