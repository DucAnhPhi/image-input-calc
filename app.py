import numpy as np
import cv2 as cv
from preprocessing import PreProcessing
from segmentation import Segmentation


class App:
    def run_with_webcam(self):
        cap = cv.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()

            # Our operations on the frame come here

            # define window of interest
            fHeight, fWidth, _ = frame.shape
            window = (100, 100, fWidth-200, fHeight-200)
            wX, wY, wWidth, wHeight = window

            # draw window
            cv.rectangle(frame, (wX, wY), (wX+wWidth,
                                           wY+wHeight), (0, 0, 255), 2)

            # crop frame according to defined window
            cropped = frame[wY:wY+wHeight, wX:wX+wWidth]

            # preprocess cropped image
            preprocessed = PreProcessing().preprocess(cropped)

            # find contours using algorithm by Suzuki et al. (1985)
            contours, hierarchy = cv.findContours(
                preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            # get bounding boxes of contours and filter them
            rects = [cv.boundingRect(cnt) for cnt in contours]
            rects = Segmentation().filter_bounding_boxes(preprocessed, rects, hierarchy)

            # draw each bounding box
            for rect in rects:
                rX, rY, rWidth, rHeight = rect
                rX = rX + wX
                rY = rY + wY
                cv.rectangle(frame, (rX, rY), (rX+rWidth,
                                               rY+rHeight), (0, 255, 0), 2)

            # Display the resulting frame
            cv.imshow('frame', frame)

            # Press ESC to quit
            if cv.waitKey(1) == 27:
                break

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run_with_img(self):
        frame = cv.imread('sample2.jpg', 1)

        # define window of interest
        fHeight, fWidth, _ = frame.shape
        window = (100, 100, fWidth-200, fHeight-200)
        wX, wY, wWidth, wHeight = window

        # draw window
        cv.rectangle(frame, (wX, wY), (wX+wWidth, wY+wHeight), (0, 0, 255), 2)

        # crop frame according to defined window
        cropped = frame[wY:wY+wHeight, wX:wX+wWidth]

        # preprocess cropped image
        preprocessed = PreProcessing().preprocess(cropped)

        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # get bounding boxes of contours and filter them
        rects = [cv.boundingRect(cnt) for cnt in contours]
        rects = Segmentation().filter_bounding_boxes(preprocessed, rects, hierarchy)

        # draw each bounding box
        for rect in rects:
            rX, rY, rWidth, rHeight = rect
            rX = rX + wX
            rY = rY + wY
            cv.rectangle(frame, (rX, rY), (rX+rWidth,
                                           rY+rHeight), (0, 255, 0), 2)

        # Display the resulting frame
        cv.imshow('frame', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    App().run_with_webcam()
    # App().run_with_img()
