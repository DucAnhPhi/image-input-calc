import cv2 as cv
import numpy as np
from contour import Contour
from fraction import Fraction
from enums import MathSign
from enums import Position


class Segmentation:
    def __init__(self, contourList, hierarchy, imgShape):
        self.contourList = contourList
        self.hierarchy = hierarchy
        self.imgShape = imgShape
        self.lineThickness = self.get_line_thickness()

    def get_line_thickness(self):
        areas = np.array([cnt.width * cnt.height for cnt in self.contourList])
        medianAreaIndices = np.argwhere(
            areas == np.percentile(areas, 50, interpolation='nearest'))
        thickness = [
            self.contourList[i[0]].get_thickness() for i in medianAreaIndices]
        # get median line thickness
        medianThickness = np.median(thickness)
        return medianThickness

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
        grouped.set_math_sign_type(MathSign.EQUAL)

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
                isNested = True
        return isNested

    def handle_nested_contour(self, cnt, index):
        parentIndex = self.hierarchy[index][-1]
        if self.is_nested_contour(cnt, parentIndex):
            parent = self.contourList[parentIndex]
            parent.add_hole(cnt.contour)
            cnt.mark_for_removal()

    def label_comma(self, currCnt, preCnt):
        yDevToCentroid = abs(currCnt.center[1] - preCnt.center[1])
        yDevToBottom = abs(currCnt.center[1] - preCnt.y2)
        if yDevToBottom < yDevToCentroid:
            if cv.contourArea(currCnt.contour) < cv.contourArea(preCnt.contour):
                currCnt.set_math_sign_type(MathSign.COMMA)

    def label_multiply(self, currCnt, preCnt):
        yDevToCentroid = abs(currCnt.center[1] - preCnt.center[1])
        yDevToTop = abs(currCnt.center[1] - preCnt.y1)
        yDevToBottom = abs(currCnt.center[1] - preCnt.y2)
        if yDevToTop > yDevToCentroid and yDevToBottom > yDevToCentroid:
            currCnt.set_math_sign_type(MathSign.MULTIPLY)

    def is_point(self, cnt):
        isPoint = cnt.trueWidth <= self.lineThickness * 2
        isPoint = isPoint and cnt.trueHeight <= self.lineThickness * 2
        return isPoint

    def label_point(self, currCnt, preCnt, postCnt):
        valid = False
        between = preCnt != None and postCnt != None
        if between:
            valid = currCnt.mathSign == None
            valid = valid and preCnt.mathSign == None
            valid = valid and postCnt.mathSign == None
            valid = valid and not self.is_point(
                preCnt) and not self.is_point(postCnt)

        if valid:
            # label points which are between two cyphers
            self.label_comma(currCnt, preCnt)
            self.label_multiply(currCnt, preCnt)
        else:
            # remove points which are not between two cyphers
            currCnt.mark_for_removal()

    def label_exponent(self, currCnt, preCnt, postCnt):
        valid = False
        before = preCnt != None
        if before:
            valid = currCnt.mathSign == None
            valid = valid and preCnt.mathSign == None
            valid = valid and not self.is_point(preCnt)
            valid = valid and cv.contourArea(
                preCnt.contour) > cv.contourArea(currCnt.contour)
            yDevToCentroid = abs(currCnt.center[1] - preCnt.center[1])
            yDevToTop = abs(currCnt.center[1] - preCnt.y1)
            valid = valid and yDevToCentroid > yDevToTop
            if valid:
                currCnt.set_position_type(Position.EXPONENT)

    def label_contours(self, lines):
        def label_helper(contours):
            for i in range(len(contours)):
                currCnt = contours[i]
                preCnt = None
                postCnt = None

                if i > 0:
                    preCnt = contours[i-1]
                if i < len(contours)-1:
                    postCnt = contours[i+1]

                if self.is_point(currCnt):
                    self.label_point(currCnt, preCnt, postCnt)
                # else:
                #     self.label_exponent(currCnt, preCnt, postCnt)

        for i in range(len(lines)):
            currLine = lines[i]
            label_helper(currLine)
            for el in currLine:
                if el.fraction != None:
                    label_helper(el.fraction.nominator)
                    label_helper(el.fraction.denominator)

        return [[cnt for cnt in line if not cnt.remove] for line in lines]

    def label_bar_type(self, cnt):
        # don't consider contours which are about to be removed
        if cnt.remove:
            return
        # only consider bars
        if not cnt.is_bar():
            return

        if cnt.is_vertical_bar():
            cnt.set_math_sign_type(MathSign.FRACTION_VERT)
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

        for tempCnt in self.contourList:
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
            cnt.set_math_sign_type(MathSign.MINUS)
        elif above and below:
            cnt.set_math_sign_type(MathSign.FRACTION_HOR)
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
            self.label_bar_type(cnt)

        fractionBars = [
            cnt for cnt in self.contourList if cnt.mathSign == MathSign.FRACTION_HOR]

        # sort fraction bars ascending by width
        fractionBars.sort(key=lambda bar: bar.width)

        # group contours to fractions starting with most narrow fraction bar
        for bar in fractionBars:
            if bar.remove:
                continue
            self.group_fraction(bar)

        # remove contours which were marked for removal before
        self.contourList = [cnt for cnt in self.contourList if not cnt.remove]

    def check_small_contours(self):
        tolerance = 0.7
        minThickness = self.lineThickness * tolerance
        # mark small contours for removal later
        for cnt in self.contourList:
            if cnt.trueWidth < minThickness or cnt.trueHeight < minThickness:
                cnt.mark_for_removal()
