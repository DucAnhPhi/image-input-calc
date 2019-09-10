import numpy as np
import cv2 as cv
from preprocessing import PreProcessing
from segmentation import Segmentation
from ordering2 import LineOrdering2
from drawing import Draw
from iic import MathSymbolClassifier
from solver import Solver
from sympy.parsing.sympy_parser import TokenError


class App:
    def __init__(self):
        self.classifier = MathSymbolClassifier('hasy_model-02.ckpt')
        self.solver = Solver()

    def solve(self, line_list):
        for line in line_list:
            line_vector = np.ndarray((len(line), 1, 32, 32))
            idx = 0
            for symbol in line:
                resized_symbol = cv.resize(symbol, (32,32))
                line_vector[idx] = resized_symbol.reshape(1, 32, 32)
            result = ''.join(self.classifier.classify(line_vector))
            print("Input:", result)
            try:
                solved = self.solver.solve(result)
                print("Solution:", solved)
            except TokenError:
                print("The given input was not recognized.")

    def process(self, frame,name="TrainingSamples/Image_"):
        preprocessed = PreProcessing().background_contour_removal(
            frame)


        #preprocessed=PreProcessing().preprocess3(frame)
        #preprocessed = cv.cvtColor(preprocessed, cv.COLOR_GRAY2BGR)
        #preprocessed = PreProcessing().background_contour_removal(preprocessed)


        preprocessedForImages=preprocessed.copy()

        print("Preprocessing Done")

        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        print("Segmentation Done")

        if (len(contours)>30): # get bounding boxes of contours and filter them
            filtered = Segmentation().filter_contours(preprocessed, contours, hierarchy)
            contoursUsedForOrdering=filtered.copy()
        else:
            contoursUsedForOrdering=contours.copy()

        # colouring preprocessing for ease in debugging
        preprocessed = cv.cvtColor(preprocessed, cv.COLOR_GRAY2BGR)

        print("Segmentation Filtering Done")

        if len(contoursUsedForOrdering)==0:
            print("ERROR NO CONTOURS DETECTED")
            cv.waitKey()
            return preprocessed


        # draw each bounding box
        #boundingBoxFrame, orderedImage = Draw().draw_bounding_boxes_around_contours(preprocessed, filteredContours)
        print("Starting Line Ordering Done")
        # create ordered List of Contours

        orderedLineList, horVec, orderedImage = LineOrdering2().get_orderedLineList2(contoursUsedForOrdering, preprocessed.copy())
        orderedImage = Draw().draw_orderedImage2(orderedLineList, horVec, orderedImage)
        print("Line Ordering Done")

        imageLineList=Segmentation().get_subimage_list_list_from_contour_list_list(preprocessedForImages,orderedLineList)
        #Segmentation().print_subimage_list_list_Images(preprocessedForImages,orderedLineList,name)

        return orderedImage, imageLineList

    def show_results(self, frame, result):

        frame = Draw().scale_image(frame, 0.25)
        result = Draw().scale_image(result , 0.25)
        cv.imshow('frame', frame)
        cv.imshow('preprocessed', result)
        # Segmentation().print_lineList_images(preprocessed,orderedLineList)


    def run_with_webcam(self):
        cap = cv.VideoCapture(0)
        while(True):

            # Capture frame-by-frame
            _, frame = cap.read()

            # Processing the frame
            preprocessed, line_list = self.process(frame)

            # Display the resulting frame
            self.show_results(frame, preprocessed)

            # Press ESC to quit
            if cv.waitKey(1) == 27:
                break

        # Perform classification
        print("Recognized symbols:")
        self.solve(line_list)
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run_with_img(self,source='sample.jpg',name="TrainingSamples/Image_"):
        frame = cv.imread(source, 1)

        preprocessed, line_list = self.process(frame,name)
        # Display the resulting frame
        self.show_results(frame, preprocessed)

        cv.waitKey(0)
        cv.destroyAllWindows()
        # Perform classification
        print("Recognized symbols:")
        self.solve(line_list)

    def run_with_video(self, file):
        cap = cv.VideoCapture(file)

        if cap.isOpened():
            while(True):

                # Capture frame-by-frame
                _, frame = cap.read()

                # Processing the frame
                preprocessed, line_list = self.process(frame)

                # Display the resulting frame
                self.show_results(frame, preprocessed)

                # Press ESC to quit
                if cv.waitKey(1) == 27:
                    break

        # Perform classification
        print("Recognized symbols:")
        self.solve(line_list)

        # When everything done,
        cap.release()
        cv.destroyAllWindows()


    def run(self, source):
        if (source == "" or source == "webcam" or source == "Webcam"):
            print("Using Webcam")
            App().run_with_webcam()

        if (source=="TrainingSamples"):
            for i in range(0, 42):
                name = ("TrainingSamples/Image_" + str(292 + i) + "_")
                source=("SampleImages\IMG_0"+str(292+i)+".JPG")
                print("Opening Image")
                print(source)
                App().run_with_img(source,name)
        sourceEnding = source.split(".", 1)[1]

        if sourceEnding == "MOV":
            print("Opening Video Clip")
            App().run_with_video(source)

        if (sourceEnding == "jpg" or sourceEnding == "JPG"):
            print("Opening Image")
            print(source)
            App().run_with_img(source)


if __name__ == '__main__':
    #App().run("TrainingSamples")#("SampleImages\IMG_0"+str(292+i)+".JPG"))#"sample.MOV")
    App().run("sample.MOV")
    # App().run_with_img()
    # App().run_with_video('sample.MOV')
