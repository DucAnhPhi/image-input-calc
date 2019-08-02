import numpy as np
import cv2
from preprocessing import PreProcessing

class Webcam:
    def show(self):
        cap = cv2.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            preprocessed = PreProcessing().preprocess(frame)

            # Display the resulting frame
            cv2.imshow('frame', preprocessed)
            
            # Press ESC to quit
            if cv2.waitKey(1) == 27:
                break
    
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Webcam().show()