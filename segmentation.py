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

    def handle_equal_bar(self, cnt):
        # build grouped contour
        bar2 = cnt.equalBar
        bX, bY, bWidth, bHeight = cv.boundingRect(cnt.contour)
        b2X, b2Y, b2Width, b2Height = cv.boundingRect(bar2.contour)
        minX = min(bX, b2X)
        maxX = max(bX+bWidth, b2X+b2Width)
        minY = min(bY, b2Y)
        maxY = max(bY+bHeight, b2Y+b2Height)
        mask = np.ones(self.imgShape[:2], dtype="uint8") * 255
        cv.rectangle(mask, (minX, minY), (maxX, maxY), (0, 0, 0), 1)
        cnts, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        grouped = Contour(cnts[1], self.imgShape, isEqualSign=True)
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
            # handle outer border
            cnt.check_outer_border()
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

    def filter(self):
        # remove contours which were marked for removal before
        self.contourList = [cnt for cnt in self.contourList if not cnt.remove]

    # def print_subimage_list_Images(self, frame, subimageList, name="Image_"):

    #     for i in range(len(subimageList)):

    #         cv.imwrite((name + str(i) + ".png"),
    #                    self.get_subimage_from_contour(frame, subimageList[i][0]))
    #         print("Saved the file")

    # def print_lineList_images(self, frame, lineList):

    #     for i in range(len(lineList)):

    #         name = str("Line_" + str(i) + "_Symbol_")
    #         self.print_subimage_list_Images(frame, lineList[i], name)

    # def print_subimage_list_list_Images(self, frame, orderedLineList, name="TrainingSamples/Image_"):

    #     for i in range(len(orderedLineList)):
    #         for j in range(len(orderedLineList[i])):
    #             cv.imwrite((name + str(i) + "_" + str(j) + ".png"),
    #                        self.get_subimage_from_contour(frame, orderedLineList[i][j]))
    #             print("Saved the file")
