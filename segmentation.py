import cv2 as cv
import numpy as np
from contour import Contour
from fraction import Fraction
from enums import BarType


class Segmentation:
    def __init__(self, contourList, hierarchy, imgShape):
        self.contourList = contourList
        self.hierarchy = hierarchy
        self.imgShape = imgShape
        self.minLineThickness = self.get_min_line_thickness()

    def get_min_line_thickness(self):
        # find top 3 biggest contours
        k = 3
        areas = np.array([cnt.width * cnt.height for cnt in self.contourList])
        n = len(areas)
        topIdx = []

        if n < 3:
            topIdx = np.array(range(n))
        else:
            topIdx = np.argpartition(areas, -k)[-k:]

        topThickness = []
        for i in topIdx:
            currentCnt = self.contourList[i]
            topThickness.append(currentCnt.get_thickness())

        if len(topThickness) == 0:
            return

        # get minimum line thickness with some tolerance
        minThickness = min(topThickness) * 0.8
        return minThickness

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

    def group_equal(self, bar1, bar2):
        # build grouped contour
        b1X, b1Y, b1Width, b1Height = cv.boundingRect(bar1.contour)
        b2X, b2Y, b2Width, b2Height = cv.boundingRect(bar2.contour)
        minX = min(b1X, b2X)
        maxX = max(b1X+b1Width, b2X+b2Width)
        minY = min(b1Y, b2Y)
        maxY = max(b1Y+b1Height, b2Y+b2Height)
        outerCnt = self.generate_contour_from_edges(minX, maxX, minY, maxY)
        grouped = Contour(outerCnt, self.imgShape)
        grouped.set_bar_type(BarType.EQUAL)

        # mark contours for removal and add new grouped contour
        bar1.mark_for_removal()
        bar2.mark_for_removal()
        self.contourList.append(grouped)

    def group_fraction(self, cnt):
        # build fraction
        fraction = Fraction(cnt, self.contourList, self.imgShape)

        # build new contour
        groupedContour = Contour(
            fraction.get_contour(), self.imgShape, fraction=fraction)

        groupedContours = [*fraction.nominator,
                           *fraction.denominator, fraction.bar]

        # mark all grouped contours for removal and add new contour to contourList
        for grouped in groupedContours:
            grouped.mark_for_removal()
        self.contourList.append(groupedContour)

    def is_nested_contour(self, cnt, parentIndex):
        isNested = False
        if parentIndex != -1:
            parent = self.contourList[parentIndex]
            if cnt.is_inside_area(parent.x1, parent.x2, parent.y1, parent.y2):
                parent.holes.append(cnt.contour)
            isNested = True
        return isNested

    def handle_nested_contour(self, cnt, index):
        parentIndex = self.hierarchy[index][-1]
        if self.is_nested_contour(cnt, parentIndex):
            parent = self.contourList[parentIndex]
            parent.add_hole(cnt.contour)
            cnt.mark_for_removal()

    def label_bar_type(self, cnt, contourList):
        # don't consider contours which are about to be removed
        if cnt.remove:
            return
        # only consider bars
        if not cnt.is_bar():
            return

        # define acceptance area of possible fraction
        minX = cnt.x1
        maxX = cnt.x2
        minY = max(cnt.y1-(cnt.radius), 0)
        maxY = cnt.y2+(cnt.radius)

        # check for contours above and below bar
        above = False
        below = False

        # remember neighbour to compose possible equal sign later
        equalBar = None
        center = ((maxY - minY) // 2) + minY

        for tempCnt in contourList:
            if tempCnt.remove:
                continue
            if cnt.contourId == tempCnt.contourId:
                continue
            if tempCnt.is_inside_area(minX, maxX, minY, center):
                above = True
                equalBar = tempCnt
                continue
            if tempCnt.is_inside_area(minX, maxX, center, maxY):
                below = True
                equalBar = tempCnt

        if not above and not below:
            cnt.set_bar_type(BarType.MINUS)
        elif above and below:
            cnt.set_bar_type(BarType.FRACTION)
        else:
            self.group_equal(cnt, equalBar)

    def group_and_classify(self):
        self.check_small_contours()

        for i in range(len(self.contourList)):
            cnt = self.contourList[i]
            # handle nested contours
            self.handle_nested_contour(cnt, i)

        for cnt in self.contourList:
            # check and label bar types
            self.label_bar_type(cnt, self.contourList)

        fractionBars = [
            cnt for cnt in self.contourList if cnt.barType == BarType.FRACTION]

        # sort fraction bars ascending by width
        fractionBars.sort(key=lambda bar: bar.width)

        # group contours to fractions starting with most narrow fraction bar
        for bar in fractionBars:
            if bar.remove:
                continue
            self.group_fraction(bar)

    def check_small_contours(self):

        # mark small contours for removal later
        for cnt in self.contourList:
            if min(cnt.trueWidth, cnt.trueHeight) < self.minLineThickness:
                cnt.mark_for_removal()

    def filter(self):
        # remove contours which were marked for removal before
        self.contourList = [cnt for cnt in self.contourList if not cnt.remove]
