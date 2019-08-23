import cv2 as cv
import numpy as np


class Draw:
    def color(self,i):
        color = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (100, 0, 255), (200, 100, 50), (10, 170, 30), (50, 90, 200),(0,70,255))
        if (i < 7):
            return color[i]
        else:
            return color[0]

    def draw_bounding_boxes(self, frame, filtered):
        rects = [cv.boundingRect(cnt) for cnt in filtered]

        for rect in rects:
            rX, rY, rWidth, rHeight = rect
            cv.rectangle(frame, (rX, rY), (rX+rWidth,
                                           rY+rHeight), (0, 255, 0), 2)
        return frame

    def draw_line(self, inIm,line, color):

        symbolList, pointList=line
        for i in range(len(symbolList)):
            cv.circle(inIm,(int(symbolList[i][1]),int(symbolList[i][2])), int(symbolList[i][3]-5),color,10)
            if(i!=0):
                cv.line(inIm,(int(symbolList[i-1][1]),int(symbolList[i-1][2])),(int(symbolList[i][1]),int(symbolList[i][2])),color,5)
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
            cv.circle(inIm,(int(orderedLine[i][1]),int(orderedLine[i][2])), int(orderedLine[i][3]-5),color,10)
            if(i!=0):
                cv.line(inIm,(int(orderedLine[i-1][1]),int(orderedLine[i-1][2])),(int(orderedLine[i][1]),int(orderedLine[i][2])),color,5)
            cv.putText(inIm, str((i+1)), (int(orderedLine[i][1]),int(orderedLine[i][2])), cv.FONT_HERSHEY_SIMPLEX, 1.0 , color,2,cv.LINE_AA)
        return inIm

    def draw_orderedImage(self,inIm,orderedLineList):
        for i in range(len(orderedLineList)):
            inIm=self.draw_orderedLine(inIm,orderedLineList[i],self.color(i))

        return inIm
