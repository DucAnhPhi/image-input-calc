import numpy as np
import cv2
from preprocessing import PreProcessing
import subimage
import orienting
import time


global LT
global FT

Commentary=True

InputSource="IMG_0289.MOV"
ShowOriginal=True
ShowProcessed=False

SubImMethod=1
#Which method would you like to use to find the SubImages
# 0 = None, 1 = ConnectedCutting, 2 = Polyline Fitting

ShowPreOriented=False

ShowLineList=True
SaveSubImages=True

Orient=True
UseBorderRemoval=True

def Commenting(Say):
    global FT
    global LT
    if(Commentary):
        if(Say=="Starting new frame"):
            print("")
            print("")

        print("           ",Say)
        print("                      Time since last Frame: ", time.time() - FT)
        print("                      Time since last Step: ", time.time() - LT)
        #print("                      This programm has been running: ", time.time() - StartTime, " seconds")
        LT = time.time()

def Processing(frame):

    global FT
    FT=time.time()
    Commenting("Starting new frame")
    #print(FrameTime," ST ", StartTime)
    FT=time.time()


    # Our operations on the frame come here
    preprocessed = PreProcessing().preprocess2(frame)
    Commenting("Preprocessing Done")
    if(SubImMethod==0 and Orient):
        print("Cannot Orient without prior parsing")
    else:
        if (UseBorderRemoval):
            BRMask = PreProcessing().BorderRemovalMask(preprocessed)
            preprocessed = np.where(BRMask == 0, preprocessed, 255)
        # Display the resulting frame
        Commenting("Borders removed")

        if (ShowOriginal):
            cv2.imshow('Original', frame)

        if (ShowProcessed):
            # cv2.imshow('Red', PreProcessing().redfilter(frame))
            # cv2.imshow('Blue', PreProcessing().bluefilter(frame))
            # cv2.imshow('Green', PreProcessing().greenfilter(frame))
            cv2.imshow('Preprocessed', preprocessed)


        if(SubImMethod==1):
            preprocessed=PreProcessing().customOpen(preprocessed)
            SubImList = subimage.ConnectedCutting(preprocessed)
            print(str(len(SubImList)), " possible symbols were found")




        if (SubImMethod == 2):
            SubImList = subimage.PolylineFitting(preprocessed)
            if(SubImList!=None):
                print(str(len(SubImList)), " possible symbols were found")

        Commenting("Divided into Subimages")
        if (Orient):
            SymbolList, PointList = orienting.Resize(SubImList.copy())
            Commenting("Resized SubImages")
            LineList,ZSIm =orienting.Orient(SymbolList,PointList,preprocessed)
            Commenting("Orient")

        if(ShowPreOriented):
            for i in range(len(SymbolList)):
                #cv2.imshow('Current SubImage Before', SubImList[i].Image)
                cv2.imshow('Current SubImage After', SymbolList[i].Image)
                cv2.waitKey(0)


        if(ShowLineList):
            cv2.imshow('PolyLine', ZSIm)
            Commenting("Oriented Symbols")
        if(SaveSubImages):
            for i in range(len(LineList)):
                for j in range(len(LineList[i])):
                    cv2.imwrite(("Line"+str(i)+"_Symbol"+str(j)+".png"), LineList[i][j].Image)
                    print("Saved the file")


class Webcam:
    def show(self):
        cap = cv2.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            Processing(frame)

            # Press ESC to quit
            if cv2.waitKey(1) == 27:
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


class Video:
    def show(self,InputSource):
        cap = cv2.VideoCapture(InputSource)

        if cap.isOpened():
            print("Input found")
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()

                Processing(frame)

                # Press ESC to quit
                if cv2.waitKey(1) == 27:
                    break

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

        else:
            print("Input not found")





if __name__ == '__main__':
    FT = time.time()
    LT = time.time()
    if(InputSource=="" or InputSource=="Webcam"):
        print("Using Webcam")
        Webcam().show()
    InputSourceEnding=InputSource.split(".", 1)[1]
    print("File type recognised as ", InputSourceEnding)
    if InputSourceEnding=="MOV":
        print("Opening Video Clip")
        Video().show(InputSource)