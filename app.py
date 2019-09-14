import numpy as np
import cv2 as cv
from preprocessing import PreProcessing
from segmentation import Segmentation
from contour import Contour
from fraction import Fraction
from solver import Solver


class App:

    def process(self, frame):
        preprocessed = PreProcessing().background_contour_removal(
            frame)
        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # initialize contour object from each contour in contour list
        contourList = [Contour(contour=cnt, imgShape=frame.shape)
                       for cnt in contours]

        # filter and group segmented contours

        # find bar types
        fractionBars = []
        equalBars = []

        for i in range(len(contourList)):
            cnt = contourList[i]
            # handle nested contours
            Segmentation().check_holes(preprocessed, contourList, hierarchy, cnt, i)
            # handle outer border
            cnt.check_outer_border()
            # check and label bar types
            cnt.check_bar_type(contourList)
            if cnt.isFractionBar:
                fractionBars.append(cnt)
            elif cnt.isEqualBar:
                equalBars.append(cnt)

        # group equal bars to single contour object
        for bar in equalBars:
            if bar.remove:
                continue
            # build grouped contour object
            bar2 = bar.equalBar
            bX, bY, bWidth, bHeight = cv.boundingRect(bar.contour)
            b2X, b2Y, b2Width, b2Height = cv.boundingRect(bar2.contour)
            minX = min(bX, b2X)
            maxX = max(bX+bWidth, b2X+b2Width)
            minY = min(bY, b2Y)
            maxY = max(bY+bHeight, b2Y+b2Height)
            mask = np.ones(frame.shape[:2], dtype="uint8") * 255
            cv.rectangle(mask, (minX, minY), (maxX, maxY), (0, 0, 0), 1)
            cnts, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            grouped = Contour(cnts[1], frame.shape, isEqualSign=True)
            # mark grouped contours for removal and add new grouped contour
            bar.remove = True
            bar2.remove = True
            contourList.append(grouped)

        # sort fraction bars ascending by width
        fractionBars.sort(key=lambda bar: bar.width)

        # group contours to fractions starting with most narrow fraction bar
        for bar in fractionBars:

            # build fraction
            fraction = Fraction(bar, contourList, frame.shape)

            # build new contour
            groupedContour = Contour(
                fraction.get_contour(), frame.shape, fraction=fraction)

            groupedContours = [*fraction.nominator,
                               *fraction.denominator, fraction.bar]

            # mark all grouped contours for removal and add new contour to contourList
            for cnt in groupedContours:
                cnt.remove = True
            contourList.append(groupedContour)

        # remove contours which were marked for removal before
        contourList = [cnt for cnt in contourList if not cnt.remove]

        cv.drawContours(
            frame, [cnt.contour for cnt in contourList], -1, (0, 255, 0), 2)

        # sort contours horizontally for now
        # TODO: replace by more sophisticated ordering
        contourList.sort(key=lambda cnt: cnt.x1)

        # unwrap nested contours and pass contour list to solver object
        unwrapped = [cnt.unwrap() for cnt in contourList]

        # derive characters and compute solution using sympy
        Solver(unwrapped)

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
    # App().run_with_webcam()
    App().run_with_img()
    # App().run_with_video('sample.MOV')
