import cv2 as cv
import numpy as np


class Segmentation:
    def filter_outer_border(self, img, cnt):
        if cnt.x1 == 0 and cnt.y1 == 0:
            if cnt.width == img.shape[1] and cnt.height == img.shape[0]:
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

    def filter_contours(self, img, contourList, hierarchy):
        filtered = []
        for i in range(len(contourList)):
            cnt = contourList[i]
            valid = self.filter_outer_border(img, cnt)
            valid = valid & self.filter_nested_contour(
                img, contourList, hierarchy, cnt, i)
            if valid:
                filtered.append(cnt)
        return filtered
