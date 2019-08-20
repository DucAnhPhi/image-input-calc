import cv2 as cv


class Segmentation:
    def filter_big_rect(self, img, rect, maxThresh=0.3):
        maxArea = len(img) * len(img[0]) * maxThresh
        _, _, rWidth, rHeight = rect
        if (rWidth*rHeight) >= maxArea:
            return False
        return True

    def filter_small_rect(self, img, rect, minThresh=100):
        _, _, rWidth, rHeight = rect
        if (rWidth*rHeight) <= minThresh:
            return False
        return True

    def filter_nested_rect(self, img, rects, hierarchy, rect, index, maxThresh=0.3):
        parentIndex = hierarchy[0][index][-1]
        if parentIndex == -1:
            return True
        else:
            pX, pY, pWidth, pHeight = rects[parentIndex]
            cX, cY, cWidth, cHeight = rect
            maxArea = len(img) * len(img[0]) * maxThresh
            if pWidth * pHeight < maxArea:
                if cX >= pX and cY >= pY:
                    if (cX + cWidth) <= (pX+pWidth) and (cY+cHeight) <= (pY+pHeight):
                        return False
            return True

    def filter_rect_on_edge(self, img, rect):
        rX, rY, rWidth, rHeight = rect
        if rX == 0 or rY == 0 or (rX+rWidth) == len(img[0]) or (rY+rHeight) == len(img):
            return False
        return True

    def filter_bounding_boxes(self, img, rects, hierarchy):
        filtered = []
        for i in range(len(rects)):
            rect = rects[i]
            valid = self.filter_big_rect(img, rect)
            valid = valid & self.filter_small_rect(img, rect)
            valid = valid & self.filter_nested_rect(
                img, rects, hierarchy, rect, i)
            valid = valid & self.filter_rect_on_edge(img, rect)
            if valid:
                filtered.append(rect)
        return filtered

    def approx_contour(self, cnt, precision=0.1):
        # approx contour shape to another shape with less # vertices depending on precision we specify - uses Douglas-Peucker algorithm.
        # epsilon is max distance from contour to approx. contour
        epsilon = precision * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        return approx
