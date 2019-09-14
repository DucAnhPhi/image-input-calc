import cv2 as cv
import numpy as np


class Segmentation:
    def check_holes(self, img, contourList, hierarchy, cnt, index, maxThresh=0.3):
        parentIndex = hierarchy[0][index][-1]
        if parentIndex == -1:
            return True
        else:
            parent = contourList[parentIndex]
            maxArea = len(img) * len(img[0]) * maxThresh
            parentArea = parent.width * parent.height
            if parentArea < maxArea:
                if cnt.is_inside_area(parent.x1, parent.x2, parent.y1, parent.y2):
                    child_area = cnt.width * cnt.height
                    if child_area > parentArea * 0.1:
                        parent.holes.append(cnt.contour)
                        cnt.remove = True
