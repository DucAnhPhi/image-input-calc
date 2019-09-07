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

            # build fraction
            fraction = Fraction(bar, contourList, frame.shape)

            # build new contour
            groupedContour = Contour(
                fraction.get_contour(), frame.shape, fraction=fraction)

            groupedContours = [*fraction.nominator,
                               *fraction.denominator, fraction.bar]

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
