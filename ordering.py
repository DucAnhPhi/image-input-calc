import cv2 as cv
from segmentation import Segmentation
import numpy as np


class OrderContours:
    def is_fraction_bar(self, frame, bar):
        bX, bY, bWidth, bHeight = cv.boundingRect(bar)
        # crop image to acceptance area of bar
        minX = int(bX)
        maxX = int(bX+bWidth)
        minY = int(
            bY-(bWidth/2))
        maxY = int(bY+(bWidth/2)+bHeight)
        acceptanceArea = frame[minY:maxY, minX:maxX]
        # create mask to remove redundant contours from acceptance area
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        cv.drawContours(mask, [bar], -1, 0, -1)
        mask = mask[minY:maxY, minX:maxX]
        acceptanceArea = cv.bitwise_not(acceptanceArea)
        acceptanceArea = cv.bitwise_and(
            acceptanceArea, acceptanceArea, mask=mask)
        acceptanceArea = cv.bitwise_not(acceptanceArea)
        acceptanceArea = cv.copyMakeBorder(
            acceptanceArea, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=[255, 255, 255])
        # find contours in acceptance area
        contours, _ = cv.findContours(
            acceptanceArea, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # filter redundant contours
        contours = [cnt for cnt in contours if Segmentation(
        ).not_outer_border(acceptanceArea, cnt)]
        # check for contours above and below bar
        center = acceptanceArea.shape[0] // 2
        above = False
        below = False
        for con in contours:
            cX, cY, cWidth, cHeight = cv.boundingRect(con)
            if cY < center:
                above = True
            else:
                below = True
            cv.rectangle(acceptanceArea, (cX, cY),
                         (cX+cWidth, cY+cHeight), (0, 255, 0), 2)
        return above and below

    def get_bars(self, contours):
        bars = []

        for cnt in contours:

            _, _, bWidth, bHeight = cv.boundingRect(cnt)
            minAreaRect = cv.minAreaRect(cnt)

            if bWidth > bHeight:
                trueWidth = max(minAreaRect[1])
                trueHeight = min(minAreaRect[1])
                if trueWidth > trueHeight * 2:
                    bars.append(cnt)

        return bars
