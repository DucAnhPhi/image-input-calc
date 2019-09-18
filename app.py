import numpy as np
import cv2 as cv

from preprocessing import PreProcessing
from segmentation import Segmentation
from contour import Contour
from fraction import Fraction
from solver import Solver
from ordering import LineOrdering
from drawing import Draw
from contour import Contour
from fraction import Fraction
from solver import Solver


class App:

    def process(self, frame, name="TrainingSamples/Image_"):
        # preprocessing
        preprocessed = PreProcessing().background_contour_removal(
            frame)

        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # limit observed contours
        if len(contours) > 500:
            contours = contours[:500]

        # initialize contour object from each contour in contour list
        contourList = [Contour(contour=cnt, imgShape=frame.shape)
                       for cnt in contours]

        # filter, classify and group segmented contours
        sg = Segmentation(contourList, hierarchy, frame.shape)
        sg.filter_small_contours()
        sg.group_and_classify()
        sg.filter()

        filtered = sg.get_contours()

        if len(filtered) == 0:
            return preprocessed

        # colouring preprocessing for ease in debugging
        preprocessed = cv.cvtColor(preprocessed, cv.COLOR_GRAY2BGR)

        cv.drawContours(
            frame, [cnt.contour for cnt in filtered], -1, (0, 255, 0), 2)

        lines = LineOrdering(filtered).get_lines(frame)

        # unwrap nested contours and pass contour list to solver object
        # derive characters and compute solution using sympy
        # solutions = [Solver([cnt.unwrap() for cnt in line]) for line in lines]

        return preprocessed  # orderedImage

    def show_results(self, frame, result):

        # frame = Draw().scale_image(frame, 0.25)
        # result = Draw().scale_image(result, 0.25)
        cv.imshow('frame', frame)
        # cv.imshow('preprocessed', result)
        # Segmentation().print_lineList_images(preprocessed,orderedLineList)

    def run_with_webcam(self):
        cap = cv.VideoCapture(0)
        while(True):

            # Capture frame-by-frame
            _, frame = cap.read()

            # Processing the frame
            preprocessed = self.process(frame)

            # Display the resulting frame
            self.show_results(frame, preprocessed)

            # Press ESC to quit
            if cv.waitKey(1) == 27:
                break

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run_with_img(self, source='sample3.jpg', name="TrainingSamples/Image_"):
        frame = cv.imread(source, 1)

        preprocessed = self.process(frame, name)

        # Display the resulting frame
        self.show_results(frame, preprocessed)

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
                self.show_results(frame, preprocessed)

                # Press ESC to quit
                if cv.waitKey(1) == 27:
                    break

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run(self, source):
        if (source == "" or source == "webcam" or source == "Webcam"):
            print("Using Webcam")
            App().run_with_webcam()

        if (source == "TrainingSamples"):
            for i in range(0, 42):
                name = ("ToClassify2/Image_" + str(292 + i) + "_")
                source = ("SampleImages\IMG_0"+str(292+i)+".JPG")
                print("Opening Image")
                print(source)
                App().run_with_img(source, name)
        sourceEnding = source.split(".", 1)[1]

        if sourceEnding == "MOV":
            print("Opening Video Clip")
            App().run_with_video(source)

        if (sourceEnding == "jpg" or sourceEnding == "JPG"):
            print("Opening Image")
            print(source)
            App().run_with_img(source)


if __name__ == '__main__':
    # App().run("TrainingSamples")#("SampleImages\IMG_0"+str(292+i)+".JPG"))#"sample.MOV")
    # App().run("sample.MOV")

    # App().run_with_webcam()
    App().run_with_img()
    # App().run_with_video('sample.MOV')
