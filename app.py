import numpy as np
import cv2 as cv
from preprocessing import PreProcessing
from segmentation import Segmentation
from contour import Contour
from fraction import Fraction


class App:

    def process(self, frame):
        preprocessed = PreProcessing().background_contour_removal(
            frame)
        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # get bounding boxes of contours and filter them
        filtered = Segmentation().filter_contours(preprocessed, contours, hierarchy)

        # initialize contour object from each contour in contour list
        contourList = [Contour(contour=cnt, imgShape=frame.shape)
                       for cnt in filtered]

        # find fraction bars
        fractionBars = [
            cnt for cnt in contourList if cnt.is_fraction_bar(contourList)]

        # sort fraction bars ascending by width
        fractionBars.sort(key=lambda bar: bar.width)

        # group contours to fractions starting with most narrow fraction bar
        for bar in fractionBars:

            # define acceptance area
            minX = bar.x1
            maxX = bar.x2
            minY = max(bar.y1-(bar.radius), 0)
            maxY = bar.y2+(bar.radius)
            center = (maxY-minY) // 2 + minY

            nominator = []
            denominator = []

            # boundary coordinates of bounding box around fraction
            newMinX = minX
            newMaxX = maxX
            newMinY = max(frame.shape)
            newMaxY = 0

            # look for contours inside acceptance area
            for nb in contourList:
                if nb.contourId == bar.contourId:
                    continue
                if nb.is_inside_area(minX, maxX, minY, center):
                    nominator.append(nb)
                    newMinX = min(newMinX, nb.x1)
                    newMaxX = max(newMaxX, nb.x2)
                    newMinY = min(newMinY, nb.y1)
                    cv.circle(frame, (nb.center[0], nb.center[1]),
                              nb.radius, (255, 0, 0), 2)
                    continue
                if nb.is_inside_area(minX, maxX, center, maxY):
                    denominator.append(nb)
                    newMinX = min(newMinX, nb.x1)
                    newMaxX = max(newMaxX, nb.x2)
                    newMaxY = max(newMaxY, nb.y2)
                    cv.circle(frame, (nb.center[0], nb.center[1]),
                              nb.radius, (255, 0, 255), 2)

            def get_contour_from_rect(shape, x1, y1, x2, y2):
                # a contour is a np.array with shape (#points, 1, 2)
                # in which each entry represents (x,y) coordinates of boundary points
                mask = np.ones(shape[:2], dtype="uint8") * 255
                cv.rectangle(mask, (x1, y1),
                             (x2, y2), (0, 0, 0), 1)
                cnts, _ = cv.findContours(
                    mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                return cnts[1]

            # build fraction
            fraction = Fraction(nominator, denominator, bar)

            # build new contour
            groupedContour = get_contour_from_rect(
                frame.shape, newMinX, newMinY, newMaxX, newMaxY)
            groupedContour = Contour(
                groupedContour, frame.shape, fraction=fraction)
            groupedContours = [*nominator, *denominator, bar]

            # remove all grouped contours and add new contour to contourList
            contourList = [
                cnt for cnt in contourList if cnt not in groupedContours]
            contourList.append(groupedContour)

            cv.drawContours(frame, [groupedContour.contour], 0, (0, 255, 0), 2)

        return preprocessed

    def run_with_webcam(self):
        cap = cv.VideoCapture(0)
        while(True):

            # Capture frame-by-frame
            _, frame = cap.read()

            # Processing the frame
            preprocessed = self.process(frame)

            # Display the resulting frame
            cv.imshow('frame', frame)

            # Press ESC to quit
            if cv.waitKey(1) == 27:
                break

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run_with_img(self):
        frame = cv.imread('sample.jpg', 1)

        preprocessed = self.process(frame)

        # Display the resulting frame
        cv.imshow('frame', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def run_with_video(self, file):
        cap = cv.VideoCapture(file)

        if cap.isOpened():
            while(True):

                # Capture frame-by-frame
                _, frame = cap.read()

                # Processing the frame
                preprocessed = self.process(frame)

                # Display the resulting frame
                # cv.imshow('frame', frame)
                cv.imshow('frame', frame)

                # Press ESC to quit
                if cv.waitKey(1) == 27:
                    break

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    App().run_with_webcam()
    # App().run_with_img()
    # App().run_with_video('sample.MOV')
