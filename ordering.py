import cv2 as cv
import numpy as np


from segmentation import Segmentation
from drawing import Draw

class LineOrdering:

    def mag(self,vectorIn):
        return np.sqrt(vectorIn[0]**2+vectorIn[1]**2)

    def get_distanceVector(self,p1,p2):
        return (p1[0]-p2[0],p1[1]-p2[1])

    def scalarVectorMul(self, vector1, scalar):
        return (vector1[0]*scalar,vector1[1]*scalar)

    def scalarMultiplication(self, vector1,vector2):
        return (vector1[0]*vector2[0]+vector1[1]*vector2[1])


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

    def get_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_coords_list(self, inList):
        p = []
        for i in range(len(inList)):
            p.append((inList[i][1], inList[i][2]))
        return p

    def relative_change(self, filteredContoursData, Bias=10):
        relativeChange = []
        for i in range(len(filteredContoursData) - 1):
            rThis = filteredContoursData[i + 1][3]
            rNext = filteredContoursData[i][3]
            relativeChange.append((rThis - rNext) / (rThis+Bias))


        return relativeChange

    def get_acceptanceRadius(self,symbolList,n):
        p= self.get_coords_list(symbolList)

        nnDistanceList=[]
        for i in range(len(symbolList)):
            nnDistance = np.Infinity
            for j in range(len(symbolList)):
                if(i!=j):
                    if(self.get_distance(p[i],p[j])<nnDistance):
                        nnDistance=self.get_distance(p[i],p[j])
            nnDistanceList.append(nnDistance)
        acceptanceRadius=n*np.median(nnDistanceList)
        return acceptanceRadius


    # naive approach
    def symbol_point_classifier(self, contours, rects):
        largeList=[]
        symbolList=[]
        pointList=[]

        largeRects=[]
        symbolRects=[]
        pointRects=[]

        areaList=[0]*len(contours)

        for i in range(len(contours)):
            areaList[i]=cv.contourArea(contours[i])

        mean = np.mean(areaList)
        for i in range(len(contours)):
            if areaList[i]>mean:
                largeList.append(contours[i])
                largeRects.append(rects[i])
            else:
                pointList.append(contours[i])
                pointRects.append(rects[i])

        largeAreaList=[0]*len(largeList)

        for i in range(len(largeList)):
            largeList[i]=cv.contourArea(largeList[i])

        largeMean=np.mean(largeAreaList)

        for i in range(len(largeList)):
            if largeAreaList[i]<5*largeMean:
                symbolList.append(largeList[i])
                symbolRects.append(largeRects[i])

        return symbolList, pointList, symbolRects, pointRects













    def determine_borders(self, filteredContoursData):
        sortedContoursData=filteredContoursData.copy()
        sortedContoursData.sort(key=lambda tup: tup[3])
        smallerContoursData=[]

        for i in range(0,int(len(sortedContoursData))):
            smallerContoursData.append(sortedContoursData[i])
        relativeChange=self.relative_change(sortedContoursData)


        firstBorder = len(relativeChange) + 2
        secondBorder = len(relativeChange) + 2

        currentMax = -1
        for i in range(len(relativeChange)):
            if(relativeChange[i]>currentMax):
                firstBorder=i
                currentMax=relativeChange[i]

        currentMax = -1
        for i in range(len(relativeChange)):
            if (relativeChange[i] > currentMax):
                if(i!=firstBorder):
                    secondBorder = i
                    currentMax = relativeChange[i]

        if(firstBorder>secondBorder):
            return firstBorder, secondBorder
        else:
            return secondBorder, firstBorder






    def symbol_point_classifier2(self, filteredContoursData):

        symbolBorder,pointBorder=self.determine_borders(filteredContoursData)

        #print(pointBorder," ", symbolBorder)

        #if we still have large background contours
        symbolBorder=len(filteredContoursData)+1
        # for debugging
        if(False):
            relativeChange = self.relative_change(filteredContoursData)
            for i in range(len(filteredContoursData)):
                if(i!=0):
                    print( filteredContoursData[i][1]," ",filteredContoursData[i][2]," ",filteredContoursData[i][3]," ",relativeChange[i-1])
                else:
                    print(filteredContoursData[i][1], " ", filteredContoursData[i][2], " ", filteredContoursData[i][3])
                if (i == pointBorder):
                    print("=======POINTS ABOVE =======")

                if (i == symbolBorder):
                    print("=======SYMBOLS ABOVE =======")

        symbolList=[]
        pointList=[]
        for i in range(len(filteredContoursData)):
            if i <= pointBorder:
                pointList.append(filteredContoursData[i])
            else:
                if i <=symbolBorder:
                    symbolList.append(filteredContoursData[i])

        if(False):
            print("Points")
            for i in range(len(pointList)):
                print(pointList[i][3])

            print("Symbols")
            for i in range(len(symbolList)):
                print(symbolList[i][3])

        return symbolList,pointList


















    def assign_first_line(self, symbolList, pointList, radiusOfAcceptance):


        newLine=[]
        newLinePoints=[]
        newLine.append(symbolList[0])
        del symbolList[0]

        addedElementToNewLine=True
        #print("length of symbolList: ", len(symbolList))
        while(addedElementToNewLine):
            addedElementToNewLine=False

            i=0

            while i <(len(newLine)):

                j = 0
                while j <len(symbolList):
                    if False:
                        print("d(",i,",",j,") = (",newLine[i][1],",",newLine[i][2],"),(",symbolList[j][1],",",
                              symbolList[j][2],") = "
                              , self.get_distance((newLine[i][1],newLine[i][2]),(symbolList[j][1],symbolList[j][2]))
                              , " radius = ",radiusOfAcceptance)
                    if self.within_radius((newLine[i][1],newLine[i][2]),(symbolList[j][1],symbolList[j][2]),radiusOfAcceptance):
                        newLine.append(symbolList[j])
                        del symbolList[j]
                        addedElementToNewLine=True

                    #print("d = ",self.get_distance((newLine[i][1],newLine[i][2]),(symbolList[j][1],symbolList[j][2])))
                    j=j+1
                i=i+1

        i = 0

        while i < (len(newLine)):

            j=0
            while j < len(pointList):
                if self.within_radius((newLine[i][1],newLine[i][2]),(pointList[j][1],pointList[j][2]),radiusOfAcceptance):
                    newLinePoints.append(pointList[j])
                    del pointList[j]
                j=j+1
            i=i+1
        line=(newLine,newLinePoints)

        return line,symbolList,pointList




    def line_assignement(self, filteredContours, inIm, n=2):

        xList, yList, rList= Segmentation().get_properties_mincircle(filteredContours)

        lineIm = inIm.copy()

        filteredContoursData=[]
        for i in range(len(filteredContours)):
            filteredContoursData.append((filteredContours[i],xList[i],yList[i],rList[i]))

        # sorting by radius
        filteredContoursData.sort(key=lambda tup: tup[3])

        symbolList, pointList = self.symbol_point_classifier2(filteredContoursData)

        for i in range(len(symbolList)):
            cv.circle(inIm, (int(symbolList[i][1]), int(symbolList[i][2])), 20, (255,255,0), 10)

        lineList=[]
        acceptanceRadius=self.get_acceptanceRadius(symbolList,n)
        while(len(symbolList)>0):
            newLine,symbolList,pointList=self.assign_first_line(symbolList,pointList,acceptanceRadius)

            lineList.append(newLine)



        #for i in range(len(symbolList)):
        #    cv.circle(lineIm,(int(symbolList[i][1]),int(symbolList[i][2])), int(symbolList[i][3]-5),(0,0,255),10)

        #for i in range(len(pointList)):
        #    cv.circle(lineIm,(int(pointList[i][1]),int(pointList[i][2])), int(pointList[i][3]),(0,255,0),2)

        lineIm=Draw().draw_lineList(lineIm,lineList)

        return lineList,lineIm



    def get_fullVectorList_and_largestVector(self, line):
        fullVectorList  =[]

        p=self.get_coords_list(line)


        for i in range(len(line)):
            for j in range(len(line)):
                if i<j:
                    fullVectorList.append(self.get_distanceVector(p[i],p[j]))

        maxLength=0
        maxLengthIndex=-1

        for i in range(len(fullVectorList)):
            if self.mag(fullVectorList[i])>maxLength:
                maxLength=self.mag(fullVectorList[i])
                maxLengthIndex=i



        return fullVectorList,fullVectorList[maxLengthIndex]



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

    def get_horVec(self,line):
        fullVectorList,largestVector=self.get_fullVectorList_and_largestVector(line)
        directedVectorList=self.directionalize(fullVectorList,largestVector)



        meanVector=self.get_meanVector(directedVectorList)


        return meanVector

    def horVec_sorter(self, line, horVec):

        newLine, newLinePoints = line


        newLine.extend(newLinePoints)

        p=self.get_coords_list(newLine)


        sortHelperTuple=[]

        for i in range(len(newLine)):
            sortHelperTuple.append((newLine[i],self.scalarMultiplication(p[i],horVec)))

        sortHelperTuple.sort(key=lambda tup: tup[1],reverse=True)

        sortedLine = []
        for i in range(len(sortHelperTuple)):
            sortedLine.append(sortHelperTuple[i][0])

        return sortedLine

    def line_ordering(self, line):

        newLine, newLinePoints = line
        horVec=self.get_horVec(newLine)

        sortedLine=self.horVec_sorter(line,horVec)

        return sortedLine


    def lineList_ordering(self, lineList):
        orderedLineList=[]
        for i in range(len(lineList)):
            if len(lineList[i][0])>2:
                orderedLineList.append(self.line_ordering(lineList[i]))
            else:
                newLine, newLinePoints = lineList[i]

                toAddLine=[]

                for i in range(len(newLine)):
                    toAddLine.append(newLine[i])

                for i in range(len(newLinePoints)):
                    toAddLine.append(newLinePoints[i])

                orderedLineList.append(toAddLine)
        orderedLineList = sorted(orderedLineList, key=len, reverse=True)


        return orderedLineList
















