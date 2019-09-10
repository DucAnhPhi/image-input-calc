import cv2 as cv
import numpy as np


from segmentation import Segmentation
from drawing import Draw

class LineOrdering2:

    def mag(self,vectorIn):
        return np.sqrt(vectorIn[0]**2+vectorIn[1]**2)

    def get_distanceVector(self,p1,p2):
        return (p2[0]-p1[0],p2[1]-p1[1])

    def scalarVectorMul(self, vector1, scalar):
        return (vector1[0]*scalar,vector1[1]*scalar)

    def scalarMultiplication(self, vector1,vector2):
        return (vector1[0]*vector2[0]+vector1[1]*vector2[1])

    def subtract(self,v1,v2):
        return (v1[0]-v2[0],v1[1]-v2[1])

    def return_radius(self, item):
        return item[3]

    def return_length(self, item):
        return len(item)


    def within_radius(self, p1, p2, r):
        x1,y1=p1
        x2,y2=p2
        if (x1-x2)**2+(y1-y2)**2>r**2:
            return False
        else:
            return True

    def turn_vector_into_intVector(self,v1):
        return (int(v1[0]),int(v1[1]))

    def get_transposed_vector(self,vector):
        return (vector[1],vector[0])

    def get_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_coords_list(self, yList,xList):
        p = []
        for i in range(len(xList)):
            p.append((yList[i], xList[i]))
        return p


    def relative_change(self, filteredContoursData, Bias=10):
        relativeChange = []
        for i in range(len(filteredContoursData) - 1):
            rThis = filteredContoursData[i + 1][3]
            rNext = filteredContoursData[i][3]
            relativeChange.append((rThis - rNext) / (rThis+Bias))


        return relativeChange

    def normalise_vector(self,v1):
        if self.mag(v1)>0:
            v1=self.scalarVectorMul(v1,(1/self.mag(v1)))
        else:
            print("Trying to normalise (0,0)")
        return v1



    def get_meanVector(self, vectorList):
        x = 0
        y = 0
        n = len(vectorList)
        for i in range(n):
            x = x + vectorList[i][0] / n
            y = y + vectorList[i][1] / n

        return (x,y)

    def directionalize(self, vectorList, direction):
        returnVectorList=[]
        for i in range(len(vectorList)):
            if self.scalarMultiplication(vectorList[i],direction) >0:
                returnVectorList.append(vectorList[i])
            else:
                returnVectorList.append(self.scalarVectorMul(vectorList[i],-1))


        return returnVectorList


    def remove_lines_with_length_less_than(self,orderedLineList, length=2):
        reducedOrderedLineList = []
        for i in range(len(orderedLineList)):
            if len(orderedLineList[i])>length:
                reducedOrderedLineList.append(orderedLineList[i])
        return reducedOrderedLineList


    def get_contoursLargerThanRadius(self,contours, cutOffRadius, rList=None):
        if rList==None:
            xList, yList, rList = Segmentation().get_properties_mincircle(contours)
        contoursLargerThanRadius=[]
        for i in range(len(contours)):
            if rList[i] >cutOffRadius:
                contoursLargerThanRadius.append(contours[i])

        return contoursLargerThanRadius














    def bordering_contours(self, contour1, contour2):
        (x1, y1), r1 = cv.minEnclosingCircle(contour1)
        (x2, y2), r2 = cv.minEnclosingCircle(contour2)

        d = self.get_distance((x1, y1), (x2, y2))

        maxAcceptRatio = 3.0
        if (d < maxAcceptRatio * r1 and d < maxAcceptRatio * r2):
            return True
        else:
            return False

    def get_largestVector(self, contours):
        xList, yList, rList = Segmentation().get_properties_mincircle(contours)

        p = []

        for i in range(len(contours)):
            p.append((yList[i], xList[i]))

        fullVectorList=[]

        for i in range(len(contours)):
            for j in range(len(contours)):
                if i<j:
                    fullVectorList.append(self.get_distanceVector(p[i],p[j]))

        maxLength=0
        maxLengthIndex=-1
        maxVector=(0.1,0)
        for i in range(len(fullVectorList)):
            if self.mag(fullVectorList[i])>maxLength:
                maxLength=self.mag(fullVectorList[i])
                maxLengthIndex=i
                maxVector=fullVectorList[maxLengthIndex]


        return maxVector


    def horVec_sorter2(self, contours, horVec):

        xList, yList, rList = Segmentation().get_properties_mincircle(contours)

        yh = horVec[0]
        xh = horVec[1]

        sortHelperTuple=[]

        for i in range(len(contours)):
            sortHelperTuple.append((contours[i],(yh*yList[i]+xh*xList[i])))

        sortHelperTuple.sort(key=lambda tup: tup[1],reverse=True)

        sortedLine = []
        for i in range(len(sortHelperTuple)):
            sortedLine.append(sortHelperTuple[i][0])

        return sortedLine

    def size_sorter(self, contours):

        xList, yList, rList = Segmentation().get_properties_mincircle(contours)



        sortHelperTuple=[]

        for i in range(len(p)):
            sortHelperTuple.append((contours[i],rList[i]))

        sortHelperTuple.sort(key=lambda tup: tup[1],reverse=True)

        sortedLine = []
        for i in range(len(sortHelperTuple)):
            sortedLine.append(sortHelperTuple[i][0])

        return sortedLine


    def get_horVec2(self, contours):


        xList, yList, rList = Segmentation().get_properties_mincircle(contours)

        p = self.get_coords_list(yList, xList)
        cutOffMean=np.mean(rList)

        vectorList = []

        for i in range(len(contours)):
            if rList[i]>cutOffMean:
                for j in range(i,len(contours)):
                    if rList[j]>cutOffMean:
                        #if self.bordering_contours(contours[i], contours[j]):
                        vectorList.append(self.get_distanceVector(p[i], p[j]))

        directionalisedVectorList=self.directionalize(vectorList,self.get_largestVector(contours))
        horVec=self.get_meanVector(directionalisedVectorList)
        print("horVec = ", horVec)
        return horVec



    def get_orthDist(self, p1,p2, vector, inIm):

        distVec=self.get_distanceVector(p1,p2)

        if(self.mag(vector)>0):
            norm=1/self.mag(vector)
        else:
            norm=1
        vector1=self.scalarVectorMul(self.get_transposed_vector(vector),norm)

        parallelLength=self.scalarMultiplication(distVec,vector1)

        parallelVec = self.scalarVectorMul(vector1, parallelLength)

        orthDistVector=self.subtract(distVec,parallelVec)

        if (np.abs(self.scalarMultiplication(orthDistVector,parallelVec))>1e-8):
            print("FATAL ERROR IN CALCULATION :" , self.scalarMultiplication(orthDistVector,parallelVec))

        orthDist= self.mag(orthDistVector)

        closestPoint=self.turn_vector_into_intVector((p1[0]+parallelVec[0], p2[1]+parallelVec[1]))


        return orthDist, inIm



    def is_accepted_in_line(self, inLineContour, candidateContour, horVec, inIm, r):

        (x1, y1), r1 = cv.minEnclosingCircle(inLineContour)
        (x2, y2), r2 = cv.minEnclosingCircle(candidateContour)


        p1=self.turn_vector_into_intVector((x1,y1))
        p2=self.turn_vector_into_intVector((x2,y2))

        distVec=self.get_distanceVector(p1,p2)


        if (self.mag(distVec) < 3 * r):
            # print(orthDist / self.mag (distVec))
            # cv.line(inIm, p1, p2, (0,0,255), 4)
            pass
        else:
            return False, inIm

        orthDist, inIm=self.get_orthDist(p1,p2,horVec,inIm)


        if (orthDist/self.mag(distVec) < 0.3 and orthDist < r and self.mag(distVec)<3*r):
            #yh = (horVec[0])/self.mag(horVec)
            #xh = (horVec[1])/self.mag(horVec)
            #print(orthDist / self.mag (distVec))
            #cv.line(inIm, p2, (int(x2+orthDist*yh),int( y2 - orthDist*xh)), (0,0,255), 10)
            return True,inIm
        else:
            return False,inIm


    def get_OrthDistList(self,contours, horVec):
        xList, yList, rList = Segmentation().get_properties_mincircle(contours)
        orthDistList=[]
        normHorVec=self.normalise_vector(horVec)

        yH=normHorVec[0]
        xH=normHorVec[1]

        xO=-yH
        yO=xH
        for i in range(len(xList)):
            yOrth=yO*yList[i]
            xOrth=xO*xList[i]
            orthDistList.append(np.sqrt(xOrth**2+yOrth**2))
        return orthDistList


    def get_linePositionList(self,orthDistList,rList):
        maxRad=0
        linePosition=0
        nearestContourList = -1
        for i in range(len(orthDistList)):
            if rList[i]>maxRad:
                linePosition=orthDistList[i]
                maxRad=rList[i]
                nearestContourList=i
        linePositionList=[]
        print(orthDistList)
        linePositionList.append(linePosition)
        return linePositionList,nearestContourList

    def get_orderedLineList2(self,contours,inIm):
        if len(contours)==0:
            print("ERROR NO CONTOURS DETECTED")
            cv.waitKey()
            return None, (0,0), inIm
        xList, yList, rList = Segmentation().get_properties_mincircle(contours)
        p=self.get_coords_list(yList,xList)

        print("Mark 2")

        cutOffRadius=0.5*np.mean(rList)
        contoursLargerThanRadius=self.get_contoursLargerThanRadius(contours, cutOffRadius,rList=rList)

        horVec= self.get_horVec2(contoursLargerThanRadius)
        horVec=(0,10)


        print("Mark 3")



        maxRad= max(rList)




        print("Mark 4")
        orthDistList=self.get_OrthDistList(contours, horVec)



        cutOffRadius = np.mean(rList)
        contoursLargerThanRadius = self.get_contoursLargerThanRadius(contours, cutOffRadius, rList=rList)
        print("Mark 4.1")
        #largeContourOrthDistList=self.get_OrthDistList(contoursLargerThanRadius, horVec)

        linePositionList,nearestContourIndex= self.get_linePositionList(orthDistList,rList)

        yh=.1*int(horVec[0])
        xh=.1*int(horVec[1])
        cv.circle(inIm, (int(xList[nearestContourIndex]), int(yList[nearestContourIndex])), int(maxRad), (255,0,0), 10)
        cv.circle(inIm, (int(xList[nearestContourIndex]), int(yList[nearestContourIndex])), int(rList[nearestContourIndex]), (0,255,0), 10)

        cv.line(inIm, (int(xList[nearestContourIndex]), int(yList[nearestContourIndex])), (int(xList[nearestContourIndex]+ xh), int(yList[nearestContourIndex]+yh)), (200, 100, 150), 10)

        for i in range(len(orthDistList)):

            cv.putText(inIm, str(int(orthDistList[i])), (int(xList[i]), int(yList[i])), cv.FONT_HERSHEY_SIMPLEX, 2 , (255,0,0),2,cv.LINE_AA)

        orderedLineList = []

        toAddLine = []


        lineAcceptanceRadius=0.5*maxRad

        print("maxRad = ", maxRad)
        print("Area of Acceptance: ", str(linePositionList[0] + lineAcceptanceRadius) , " to ", str(linePositionList[0] - lineAcceptanceRadius))


        contoursCopy = contours.copy()
        for i in range(len(linePositionList)):

            #print("linePositionList: ", i, " / ", linePositionList[i])
            for j in range(len(contoursCopy)):
                #print("orthDistList: ", j, " / ", orthDistList[j])
                if orthDistList[j]< (linePositionList[i] + lineAcceptanceRadius) and orthDistList[j] > (linePositionList[i] - lineAcceptanceRadius):
                    toAddLine.append(contours[j])
                    print("Added a contour")
                else:
                    print(" ")
            orderedLineList.append(toAddLine)


        #while len(contoursCopy)>0:
        #    toAddLine=[]
        #    toAddLine.append(contoursCopy[0])
        #    del contoursCopy[0]
        #    i=0
        #    while i<len(toAddLine):
        #        j=0
        #        (x, y), r = cv.minEnclosingCircle(toAddLine[i])
        #        if r>cutOffRadius:
        #            while j<len(contoursCopy):
        #                isAccepted, inIm = self.is_accepted_in_line(toAddLine[i],contoursCopy[j],horVec,inIm,maxRad)
        #                if isAccepted:
        #                    toAddLine.append(contoursCopy[j])
        #                    del contoursCopy[j]
        #                else:
        #                    j=j+1
        #        i=i+1

        #    sortedToAddLine=self.horVec_sorter2(toAddLine,horVec)
        #    orderedLineList.append(sortedToAddLine)


        print("Mark 5")
        reducedOrderedLineList=self.remove_lines_with_length_less_than(orderedLineList)


        print("Mark 6")
        for i in range(len(reducedOrderedLineList)):
            reducedOrderedLineList[i]=self.horVec_sorter2(reducedOrderedLineList[i],horVec)



        horVec = self.normalise_vector(horVec)
        print("Mark 7")



        return reducedOrderedLineList, horVec, inIm



