import cv2 as cv
import numpy as np
from contour import Contour
from fraction import Fraction


class Segmentation:
    def __init__(self, contourList, hierarchy, imgShape):
        self.contourList = contourList
        self.hierarchy = hierarchy
        self.imgShape = imgShape

    def get_contours(self):
        return self.contourList

    def generate_contour_from_edges(self, x1, x2, y1, y2):
        top = []
        bottom = []
        left = []
        right = []
        for i in range(x2-x1-1):
            if i == 0:
                continue
            top.append([x1 + i, y1])
            bottom.append([x2 - i, y2])
        for i in range(y2-y1-1):
            if i == 0:
                continue
            right.append([x2, y1 + i])
            left.append([x1, y2 - i])
        contour = [*top, *right, *bottom, *left]
        contour = np.array(contour).astype(np.int32)
        return contour

    def handle_equal_bar(self, cnt):
        # build grouped contour
        bar2 = cnt.equalBar
        bX, bY, bWidth, bHeight = cv.boundingRect(cnt.contour)
        b2X, b2Y, b2Width, b2Height = cv.boundingRect(bar2.contour)
        minX = min(bX, b2X)
        maxX = max(bX+bWidth, b2X+b2Width)
        minY = min(bY, b2Y)
        maxY = max(bY+bHeight, b2Y+b2Height)
        outerCnt = self.generate_contour_from_edges(minX, maxX, minY, maxY)
        grouped = Contour(outerCnt, self.imgShape, isEqualSign=True)
        # mark contours for removal and add new grouped contour
        cnt.remove = True
        bar2.remove = True
        self.contourList.append(grouped)

    def handle_fraction_bar(self, cnt):
        # build fraction
        fraction = Fraction(cnt, self.contourList, self.imgShape)

        # build new contour
        groupedContour = Contour(
            fraction.get_contour(), self.imgShape, fraction=fraction)

        groupedContours = [*fraction.nominator,
                           *fraction.denominator, fraction.bar]

        # mark all grouped contours for removal and add new contour to contourList
        for cnt in groupedContours:
            cnt.remove = True
        self.contourList.append(groupedContour)

    def group_and_classify(self):
        # find bar types
        equalBars = []
        fractionBars = []

        for i in range(len(self.contourList)):
            cnt = self.contourList[i]
            # handle nested contours
            cnt.check_holes(self.contourList, self.hierarchy, cnt, i)
            # check and label bar types
            cnt.check_bar_type(self.contourList)
            if cnt.isFractionBar:
                fractionBars.append(cnt)
            elif cnt.isEqualBar:
                equalBars.append(cnt)

        # group equal bars to single contour object
        for bar in equalBars:
            if bar.remove:
                continue
            self.handle_equal_bar(bar)

        # sort fraction bars ascending by width
        fractionBars.sort(key=lambda bar: bar.width)

        # group contours to fractions starting with most narrow fraction bar
        for bar in fractionBars:
            if bar.remove:
                continue
            self.handle_fraction_bar(bar)

    def filter_small_contours(self):
        # find top 3 biggest contours without outer frame border
        k = 4
        areas = np.array([cnt.width * cnt.height for cnt in self.contourList])
        n = len(areas)
        topIdx = []

        if n < 4:
            topIdx = np.array(range(n))
        else:
            topIdx = np.argpartition(areas, -k)[-k:]

        topThickness = [self.contourList[i].get_thickness(
        ) for i in topIdx if not self.contourList[i].is_outer_border()]

        if len(topThickness) == 0:
            return

        # get minimum line thickness
        minThickness = min(topThickness)

        # get minimum area with some tolerance
        minArea = (minThickness ** 2) * 0.8
        
        print("MinArea: ",minArea)

        # filter too small contours
        self.contourList = [
            cnt for cnt in self.contourList if cnt.width * cnt.height > minArea]
        
        for cnt in self.contourList:
            print("Area: ",cnt.width * cnt.height)

    def filter(self):
        # remove contours which were marked for removal before
        self.contourList = [cnt for cnt in self.contourList if not cnt.remove]
