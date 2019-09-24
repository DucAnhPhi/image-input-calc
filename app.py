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
    def __init__(self, solver):
        self.solver = solver

    def process(self, frame, name="TrainingSamples/Image_"):
        # preprocessing for contour detection
        preprocessed = PreProcessing().background_contour_removal(
            frame)

        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # limit observed contours
        if len(contours) > 500:
            contours = contours[:500]

        # ignore first contour, as it is outer border of the frame
        contours = contours[1:]
        hierarchy = hierarchy[0][1:]-1
        hierarchy = np.where(hierarchy < 0, -1, hierarchy)

        if len(contours) == 0:
            return preprocessed

        # initialize contour object from each contour in contour list
        binarized = PreProcessing().custom_binarize(frame)
        contourList = [Contour(contour=cnt, imgShape=frame.shape, frameBinary=binarized)
                       for cnt in contours]

        # filter, classify and group segmented contours
        sg = Segmentation(contourList, hierarchy, frame.shape)
        sg.group_and_classify()

        filtered = sg.get_contours()

        if len(filtered) == 0:
            return preprocessed

        # colouring preprocessing for ease in debugging
        preprocessed = cv.cvtColor(preprocessed, cv.COLOR_GRAY2BGR)

        lines = LineOrdering(filtered).get_lines(frame)

        # label contours with additional positional information
        lines = sg.label_contours(lines)

        for l in range(len(lines)):
            line = lines[l]
            for i in range(len(line)):
                cnt = line[i]
                cv.putText(frame, str(l) + str(i), (cnt.center[0], cnt.center[1]),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        solutions = [self.solver.solve([cnt.unwrap() for cnt in line], frame)
                     for line in lines if len(line) > 2]

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

    # def run(self, source):
    #     if (source == "" or source == "webcam" or source == "Webcam"):
    #         print("Using Webcam")
    #         App().run_with_webcam()

    #     if (source == "TrainingSamples"):
    #         for i in range(0, 42):
    #             name = ("ToClassify2/Image_" + str(292 + i) + "_")
    #             source = ("SampleImages\IMG_0"+str(292+i)+".JPG")
    #             print("Opening Image")
    #             print(source)
    #             App().run_with_img(source, name)
    #     sourceEnding = source.split(".", 1)[1]

    #     if sourceEnding == "MOV":
    #         print("Opening Video Clip")
    #         App().run_with_video(source)

    #     if (sourceEnding == "jpg" or sourceEnding == "JPG"):
    #         print("Opening Image")
    #         print(source)
    #         App().run_with_img(source)


if __name__ == '__main__':
    solver = Solver()
    # App().run("TrainingSamples")#("SampleImages\IMG_0"+str(292+i)+".JPG"))#"sample.MOV")
    # App().run("sample.MOV")

    # App(solver).run_with_webcam()
    App(solver).run_with_img()
    # App(solver).run_with_video('sample.MOV')
