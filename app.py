import numpy as np
import cv2 as cv
from preprocessing import PreProcessing
from segmentation import Segmentation


class App:
    def process(self, frame):
        # preprocess cropped image
        preprocessed = PreProcessing().preprocess(frame)

        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # get bounding boxes of contours and filter them
        filtered = Segmentation().filter_contours(preprocessed, contours, hierarchy)
        rects = [cv.boundingRect(cnt) for cnt in filtered]

        # draw each bounding box
        for rect in rects:
            rX, rY, rWidth, rHeight = rect
            cv.rectangle(frame, (rX, rY), (rX+rWidth,
                                           rY+rHeight), (0, 255, 0), 2)
        # BorderRemoval()
        # Segmentation()
        # LineFitting()
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
    # App().run_with_img()
    App().run_with_video('sample.MOV')
