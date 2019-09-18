import cv2 as cv
import numpy as np

from contour import Contour

class Draw:
    def color(self,i):
        color = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (100, 0, 255), (200, 100, 50), (10, 170, 30), (50, 90, 200),(0,70,255))
        if (i < 7):
            return color[i]
        else:
            return color[0]

    def draw_bounding_boxes_around_contours(self, frame, filtered):
        rects = [cv.boundingRect(cnt) for cnt in filtered]

        for rect in rects:
            rX, rY, rWidth, rHeight = rect
            cv.rectangle(frame, (rX, rY), (rX+rWidth,
                                           rY+rHeight),(0, 255, 0), 2)
        return frame

    def draw_line(self, inIm,line, color):

        symbolList, pointList=line
        for i in range(len(symbolList)):
            cv.circle(inIm,(int(symbolList[i][1]),int(symbolList[i][2])), int(symbolList[i][3]),color,2)
            if(i!=0):
                cv.line(inIm,(int(symbolList[i-1][1]),int(symbolList[i-1][2])),(int(symbolList[i][1]),int(symbolList[i][2])),color,2)
            cv.putText(inIm, str((i+1)), (int(symbolList[i][1]),int(symbolList[i][2])), cv.FONT_HERSHEY_SIMPLEX, 1.0 , color,2,cv.LINE_AA)
        for i in range(len(pointList)):
            cv.circle(inIm,(int(pointList[i][1]),int(pointList[i][2])), int(pointList[i][3]),color,2)
        return inIm

    def draw_lineList(self,inIm,lineList):

        for i in range(len(lineList)):

            inIm=self.draw_line(inIm,lineList[i],self.color(i))

        return inIm


    def draw_orderedLine(self, inIm,orderedLine,color):
        for i in range(len(orderedLine)):
            cv.circle(inIm,(int(orderedLine[i][1]),int(orderedLine[i][2])), int(orderedLine[i][3]),color,10)
            if(i!=0 ):
                cv.line(inIm,(int(orderedLine[i-1][1]),int(orderedLine[i-1][2])),(int(orderedLine[i][1]),int(orderedLine[i][2])),color,5)
            cv.putText(inIm, str((i+1)), (int(orderedLine[i][1]),int(orderedLine[i][2])), cv.FONT_HERSHEY_SIMPLEX, 1.0 , color,2,cv.LINE_AA)
        return inIm

    def draw_orderedImage(self,inIm,orderedLineList):
        for i in range(len(orderedLineList)):
            inIm=self.draw_orderedLine(inIm,orderedLineList[i],self.color(i))

        return inIm







    def draw_orderedLine2(self, inIm,orderedLine,color,horVec):
        xList = []
        yList = []
        rList = []


        yh=int(100*horVec[0])
        xh=int(100*horVec[1])


        for i in range(len(orderedLine)):
            (x, y), r = cv.minEnclosingCircle(orderedLine[i])

            xList.append(int(x))
            yList.append(int(y))
            rList.append(int(r))


        for i in range(len(orderedLine)):
            cv.circle(inIm,(xList[i],yList[i]), rList[i],color,10)
            cv.line(inIm, (xList[i], yList[i]), (xList[i]+xh, yList[i]+yh), (200,100,150), 10)
            if(i!=0 ):
                cv.line(inIm,(xList[i-1],yList[i-1]),(xList[i],yList[i]),color,10)
            #cv.putText(inIm, str((i+1)), (xList[i],yList[i]), cv.FONT_HERSHEY_SIMPLEX, 2 , color,1,cv.LINE_AA)
        return inIm

    def draw_orderedImage2(self,orderedLineList,horVec,inIm):
        for i in range(len(orderedLineList)):
            inIm=self.draw_orderedLine2(inIm,orderedLineList[i],self.color(i),horVec)

        return inIm









    def scale_image(self,inIm, scale):
        width = int(inIm.shape[1] * scale)
        height = int(inIm.shape[0] * scale)
        dim=(width,height)
        retIm = cv.resize(inIm, dim, cv.INTER_AREA)

        return retIm


    def draw_contours(self,contours,inIm):
        for i in range(len(contours)):
            (x, y), r = cv.minEnclosingCircle(contours[i])
            cv.circle(inIm,(int(x),int(y)), 5,(100,0,0),2)
        return inIm

    # Stolen from Segmentation.py Importing it here seemed a BAD idea
    def get_properties_mincircle(self,filteredContours):
        xList = []
        yList = []
        rList = []

        for i in range(len(filteredContours)):

            (x, y), r = cv.minEnclosingCircle(filteredContours[i])

            xList.append(x)
            yList.append(y)
            rList.append(r)

        return xList,yList,rList


    def LineFeedBack(self,inIm,contours,nearestContour,orthDistList,horVec,maxRad):

        xList, yList, rList = self.get_properties_mincircle(contours)
        (xNear, yNear), rNear = cv.minEnclosingCircle(nearestContour)

        intXNear=int(xNear)
        intYNear=int(yNear)
        intRNear=int(rNear)

        yh=.1*int(horVec[0])
        xh=.1*int(horVec[1])
        cv.circle(inIm, (intXNear,intYNear), int(maxRad), (255,0,0), 10)
        cv.circle(inIm, (intXNear,intYNear), intRNear, (0,255,0), 10)

        cv.line(inIm, (intXNear,intYNear), (int(xNear+ xh), int(yNear+yh)), (200, 100, 150), 10)

        for i in range(len(orthDistList)):

            cv.putText(inIm, str(int(orthDistList[i])), (int(xList[i]), int(yList[i])), cv.FONT_HERSHEY_SIMPLEX, 2 , (255,0,0),2,cv.LINE_AA)


        return inIm

    
    
    def print_lines(self, lines, frame, name="ToClassify/Test_Line_"):
        images=[]
        for i in range(len(lines)):
            
            Images = [cnt.get_subimage() for cnt in lines[i]]
            for j in range(len(Images)):
                cv.imwrite((name + str(i) + "_"+ "Symbol"+ str(j) + ".png"), Images[j])
                print("Saved image "+(name + str(i) + "_"+ "Symbol"+ str(j) + ".png"))

