import numpy as np
import cv2 as cv

from preprocessing import PreProcessing

from segmentation import Segmentation

from ordering import LineOrdering
from ordering2 import LineOrdering2

from drawing import Draw





class App:

    def line_ordering(self, contours, inIm, method=1):

        if method ==1:
            lineList, orderedImage = LineOrdering().line_assignement(contours, inIm, n=2)

            orderedLineList = LineOrdering().lineList_ordering(lineList)

            orderedImage = Draw().draw_orderedImage(inIm, orderedLineList)
        if method ==2:
            orderedLineList, horVec, orderedImage = LineOrdering2().get_orderedLineList2(contours,inIm)

            orderedImage = Draw().draw_orderedImage2( orderedLineList, horVec,orderedImage)

        return orderedLineList,orderedImage

    def process(self, frame):
        preprocessed = PreProcessing().background_contour_removal(
            frame)
        # find contours using algorithm by Suzuki et al. (1985)
        contours, hierarchy = cv.findContours(
            preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        print("Preprocessing Done")

        # get bounding boxes of contours and filter them
        filtered = Segmentation().filter_contours(preprocessed, contours, hierarchy)

        print("Segmentation Done")

        # creating a List of the Filtered Contours
        filteredContours=Segmentation().filtered_to_contourlist(filtered)

        print("Segmentation Filtering Done")

        # colouring preprocessing for ease in debugging
        preprocessed = cv.cvtColor(preprocessed, cv.COLOR_GRAY2BGR)


        # draw each bounding box
        #boundingBoxFrame, orderedImage = Draw().draw_bounding_boxes_around_contours(preprocessed, filteredContours)
        print("Starting Line Ordering Done")
        # create ordered List of Contours
        orderedLineList, orderedImage = self.line_ordering(filteredContours, preprocessed.copy(), method=2)
        print("Line Ordering Done")

        return orderedImage

    def show_results(self, frame, result):

        frame = Draw().scale_image(frame, 0.25)
        result = Draw().scale_image(result , 0.25)
        cv.imshow('frame', frame)
        cv.imshow('preprocessed', result)
        cv.waitKey()
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

    def run_with_img(self,source='sample.jpg'):
        frame = cv.imread(source, 1)

        preprocessed = self.process(frame)

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

        sourceEnding = source.split(".", 1)[1]

        if sourceEnding == "MOV":
            print("Opening Video Clip")
            App().run_with_video(source)

        if (sourceEnding == "jpg" or sourceEnding == "JPG"):
            print("Opening Image")
            print(source)
            App().run_with_img(source)


if __name__ == '__main__':
    App().run("SampleImages\IMG_0328.JPG")#"sample.MOV")
    # App().run_with_webcam()
    # App().run_with_img()
    # App().run_with_video('sample.MOV')
