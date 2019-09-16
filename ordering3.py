import cv2 as cv
import numpy as np


from segmentation import Segmentation
from drawing import Draw

from contour import Contour

from fraction import Fraction


class LineOrdering3:

    # returns the magnitude of a vector
    def mag(self,vectorIn):
        return np.sqrt(vectorIn[0]**2+vectorIn[1]**2)

    # returns the distance between two points
    def get_distanceVector(self,p1,p2):
        return (p2[0]-p1[0],p2[1]-p1[1])

    # returns the vector multiplied with the scalar
    def scalarVectorMul(self, vector, scalar):
        return (vector[0]*scalar,vector[1]*scalar)

    # returns the scalar product of two vectors
    def scalarMultiplication(self, vector1,vector2):
        return (vector1[0]*vector2[0]+vector1[1]*vector2[1])

    # returns the vector difference between two vectors
    def subtract(self,v1,v2):
        return (v1[0]-v2[0],v1[1]-v2[1])

    # returns the integerized vector of a vector. Important for pixels
    def turn_vector_into_intVector(self,v1):
        return (int(v1[0]),int(v1[1]))

    # swap x and y. Usefull for transition of (x,y) and (y,x) vectors
    def get_transposed_vector(self,vector):
        return (vector[1],vector[0])

    #get distance between two points
    def get_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # transfer y- and x- coordinate-Lists to a List of (y,x)-coordinate tupels
    def get_coords_list(self, yList,xList):
        p = []
        for i in range(len(xList)):
            p.append((yList[i], xList[i]))
        return p

    # normalise a vector. This gives an error if you try to normalise (0,0)
    def normalise_vector(self,v1):
        if np.linalg.norm(v1)>0:
            v1=self.scalarVectorMul(v1,(1/np.linalg.norm(v1)))
        else:
            print("Trying to normalise (0,0)")
        return v1

    # get the average Vector out of a List of Vectors
    def get_meanVector(self, vectorList):
        if(len(vectorList)==0):
            return (0,0)

        (x,y)=np.sum(vectorList, axis=0)/len(vectorList)
        return (x,y)

    def get_medianVector(self, vectorList,largestVector,acceptanceAngle=360):
        largestVectorAngle=np.arccos((largestVector[1])/np.sqrt((largestVector[1])**2+(largestVector[0])**2+1e-6))
        if (len(vectorList) == 0):
            return (0, 0)
        xList=[]
        yList=[]
        for i in range(len(vectorList)):
            x=vectorList[i][0]
            y=vectorList[i][1]
            angle= np.arccos((y)/np.sqrt(y*y+x*x+1e-6))

            if np.abs(angle-largestVectorAngle) < acceptanceAngle * np.pi/180:
                xList.append(x)
                yList.append(y)

        xMed=np.median(xList)
        yMed=np.median(yList)
        return (xMed,yMed)

    # reverse all Vectors in a List that point in the opposite direction as "direction"
    def directionalize(self, vectorList, direction):
        returnVectorList=[]
        for i in range(len(vectorList)):
            if self.scalarMultiplication(vectorList[i],direction) >0:
                returnVectorList.append(vectorList[i])
            else:
                returnVectorList.append(self.scalarVectorMul(vectorList[i],-1))


        return returnVectorList

    # Input is a List of Lists. This removes all Sub-Lists with length than 2 from the Input-List
    def remove_lines_with_length_less_than(self,orderedLineList, length=2):
        reducedOrderedLineList = []
        for i in range(len(orderedLineList)):
            if len(orderedLineList[i])>length:
                reducedOrderedLineList.append(orderedLineList[i])
        return reducedOrderedLineList


    def remove_lines_with_bordering_larger_lines(self,orderedLineList):
        emptyLine=[]
        n=len(orderedLineList)
        for i in range(n-1):
            if len(orderedLineList[i])<=len(orderedLineList[i+1]):
                orderedLineList[i]=emptyLine
        for i in range(n - 1):
            j=n-i-1
            if len(orderedLineList[j]) <= len(orderedLineList[j-1]):
                orderedLineList[j] = emptyLine
        return orderedLineList

    #sort the contours by position in horVec direction
    def horVec_sorter(self, contourList, horVec):
        
        sortedLine = sorted(contourList,key = lambda cnt: cnt.horDist)

        return sortedLine



    # get the index of the maximum value in a list
    def get_index_with_max_value(self,anyList):
        if len(anyList)==0:
            print("ERROR: Empty List was given")
            return -1
        minIndex=0
        minValue=anyList[0]
        for i in range(len(anyList)):
            if anyList[i]<minValue:
                minValue=anyList[i]
                minIndex=i
        return minIndex



    # get the orthogonal distance to (0,0) of every contour
    def get_OrthDistList(self,contourList, horVec):
        #determining some variables to be used later
        xList = list(cnt.x for cnt in contourList)
        yList = list(cnt.y for cnt in contourList)
        rList = list(cnt.radius for cnt in contourList)

        orthDistList=[]
        normHorVec=self.normalise_vector(horVec)

        yH=normHorVec[0]
        xH=normHorVec[1]

        xO=-yH
        yO=xH
        for i in range(len(xList)):
            yOrth=yO*yList[i]
            xOrth=xO*xList[i]
            contourList[i].orthDist=(np.sqrt(xOrth**2+yOrth**2))

            yHor=yH*yList[i]
            xHor=xH*xList[i]
            contourList[i].horDist=(np.sqrt(xHor**2+yHor**2))






    # This is an arbitrary acceptance condition which may be used for calculating the horizontal Vector of a line.
    def bordering_contours(self, contour1, contour2):
        (x1, y1), r1 = cv.minEnclosingCircle(contour1)
        (x2, y2), r2 = cv.minEnclosingCircle(contour2)

        d = self.get_distance((x1, y1), (x2, y2))

        maxAcceptRatio = 3.0
        if (d < maxAcceptRatio * r1 and d < maxAcceptRatio * r2):
            return True
        else:
            return False

    #Here we get a list of all vectors between any contours (fullVectorList) and a list of all vectors larger than a certain radius (largeContourVectorList).
    # We need this list to determine the horizontal Vector
    def get_fullVectorList_and_largeContourVectorList(self,rList,points,cutOffRadius):

        fullVectorList = []
        largeContourVectorList = []

        # get a List of all Vectors
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distanceVector=self.get_distanceVector(points[i], points[j])
                fullVectorList.append(distanceVector)
                if rList[i] > cutOffRadius:
                    if rList[j] > cutOffRadius:
                        largeContourVectorList.append(distanceVector)

        return fullVectorList,largeContourVectorList

    # returns the distance vector of the two most distant contours in arbitrary direction. This is a good vector to use for directionalisation
    def get_largestVector(self, fullVectorList):

        maxLength=0
        maxLengthIndex=-1
        maxVector=(0.1,0)

        for i in range(len(fullVectorList)):
            if np.linalg.norm(fullVectorList[i])>maxLength:
                maxLength=np.linalg.norm(fullVectorList[i])
                maxLengthIndex=i
                maxVector=fullVectorList[maxLengthIndex]

        print(maxVector)

        maxVector=list(maxVector)

        if maxVector[1]<0:
            maxVector[0]=-maxVector[0]
            maxVector[1]=-maxVector[1]
        

        return maxVector



    # Here we determine the direction of the line of text.
    def get_horVec2(self, contourList):
        # Concept:
        # We want to determine the direction in which the line is written.
        # The Problem is that our Input is likely to not be perfect.
        # Very likely we will have very many random dots and some large contours which do not belong into the line.
        # For determining the horizontal Vector (horVec) we will use a list of all possible vectors pointing from one contour to another.
        # Because a significant percentage of these will be between in-line variables, we can determine the horVec by calculating the mean.
        #
        # ToDo: There could be a problem if the largestVector is orthogonal to the line.
        #



        #Getting some general variables for latter usage
        xList = list(cnt.x for cnt in contourList)
        yList = list(cnt.y for cnt in contourList)
        rList = list(cnt.radius for cnt in contourList)
        points = self.get_coords_list(yList, xList)

        # we calculate here a radius that should be ideally larger than the points and smaller than the symbols.
        cutOffRadius=np.mean(rList)/2

        # We get the full Vector list here (and the one of Vectors between contours larger than cutOffRadius, which is actually a lot more usefull)
        fullVectorList,largeContourVectorList=self.get_fullVectorList_and_largeContourVectorList(rList,points,cutOffRadius)



        # We directionalise with the help of largest Vector. This Vector is not likely parallel to the horVec.
        # However, it will likely not be close to orthogonal to it, which is important in this case.
        largestVector=self.get_largestVector(largeContourVectorList)
        directionalisedVectorList=self.directionalize(fullVectorList,largestVector) #fullVectorList))

        # We normalise the Vectors, because far of contours produce large Vectors. By normalising all Vectors, we reduce this problem.
        # The process also works without. But this is an improvement
        print("B: We found ",len(directionalisedVectorList), " vectors")
        for i in range(len(directionalisedVectorList)):
            directionalisedVectorList[i]=self.normalise_vector(directionalisedVectorList[i])

        # We calculate the median Vector
        horVec=self.get_medianVector(directionalisedVectorList,largestVector, acceptanceAngle=45)


        #to avoid it getting integerized out of existance
        horVec=self.scalarVectorMul(horVec,100)



        print("horVec = ", horVec)

        horAngle= np.arccos((horVec[1])/np.sqrt(horVec[1]**2+horVec[0]**2+1e-6)) *180/np.pi

        print("TextAngle = ", horAngle, " degrees")

        return horVec








    # get a list of the position of all lines in orthogonal Distance.
    def get_linePositionList(self,contourList,lineAcceptanceRadius):
        
        orthDistList = list(cnt.orthDist for cnt in contourList)
        minOrthDist=np.min(orthDistList)
        maxOrthDist=np.max(orthDistList)

        linePositionList=[]

        currentPoint=minOrthDist

        while currentPoint < maxOrthDist:
            linePositionList.append(currentPoint)
            currentPoint=currentPoint+lineAcceptanceRadius*0.4

        return linePositionList


    def get_lineRad(self,rList):
        rListCopy=rList.copy()

        rListCopy.sort(reverse=True)

        for i in range(len(rListCopy)-1):
            if rListCopy[i]<2*rListCopy[i+1]:
                return 0.5*rListCopy[i]

        return 0.5*rListCopy[0]



    def get_orderedLineList3(self,contourList,inIm):
        # check if contours were detected. If not, nothing can be done here, except for errors produced
        if len(contourList)==0:
            print("ERROR NO CONTOURS DETECTED")
            cv.waitKey()
            return None, (0,0), inIm

        #determining some variables to be used later
        xList = list(cnt.x for cnt in contourList)
        yList = list(cnt.y for cnt in contourList)
        rList = list(cnt.radius for cnt in contourList)


        maxRad = max(rList)

        cutOffRadius = np.mean(rList)
        lineAcceptanceRadius=self.get_lineRad(rList)

        # get the direction in which the line is written
        horVec= self.get_horVec2(contourList)

        # determine the distance from (0,0) have along the orthogonal axis to the writing direction
        self.get_OrthDistList(contourList, horVec)

        # determine the position of all the lines in the image
        linePositionList= self.get_linePositionList(contourList,lineAcceptanceRadius)

        # Just feedback
        #inIm = Draw().LineFeedBack(inIm,contourList,contourList[0],orthDistList,horVec,maxRad)



        # We will return this. Time to fill it
        orderedLineList = []
        # We add the lines separately
        toAddLine = []

        # print("maxRad = ", maxRad)
        # print("Area of Acceptance: ", str(linePositionList[0] + lineAcceptanceRadius) , " to ", str(linePositionList[0] - lineAcceptanceRadius))

        # In case you want to change it
        contoursUsed = contourList.copy()

        # For every line we check every contour if it fits.
        for i in range(len(linePositionList)):
            toAddLine=[]
            for j in range(len(contoursUsed)):
                if contoursUsed[j].orthDist< (linePositionList[i] + lineAcceptanceRadius) and contoursUsed[j].orthDist > (linePositionList[i] - lineAcceptanceRadius):
                    toAddLine.append(contoursUsed[j])
            # Line is checked. Time to add it.
            orderedLineList.append(toAddLine)

        # remove Lines with neighbouring lines that are longer
        maximumOrderedLineList=self.remove_lines_with_bordering_larger_lines(orderedLineList)

        # remove Lines with length less than 2
        reducedOrderedLineList=self.remove_lines_with_length_less_than(maximumOrderedLineList)

        # sort all lines along the horVec axis
        for i in range(len(reducedOrderedLineList)):
            reducedOrderedLineList[i]=self.horVec_sorter(reducedOrderedLineList[i],horVec)


        # not necessary, but nice to do.
        horVec = self.normalise_vector(horVec)


        print("We have ", len(reducedOrderedLineList), " lines")

        for i in range(len(reducedOrderedLineList)):
            print("   Line: ", i, " has ", len(reducedOrderedLineList[i]), " symbols")
        return reducedOrderedLineList, horVec, inIm


























    def compressFractions(self,contourList, frame):
                # find bar types
        fractionBars = []
        equalBars = []

        for cnt in contourList:
            # check and label bar types
            cnt.check_bar_type(contourList)
            if cnt.isFractionBar:
                fractionBars.append(cnt)
            elif cnt.isEqualBar:
                equalBars.append(cnt)

        # group equal bars to single contour object
        for bar in equalBars:
            if bar.remove:
                continue
            # build grouped contour object
            bar2 = bar.equalBar
            bX, bY, bWidth, bHeight = cv.boundingRect(bar.contour)
            b2X, b2Y, b2Width, b2Height = cv.boundingRect(bar2.contour)
            minX = min(bX, b2X)
            maxX = max(bX+bWidth, b2X+b2Width)
            minY = min(bY, b2Y)
            maxY = max(bY+bHeight, b2Y+b2Height)
            mask = np.ones(frame.shape[:2], dtype="uint8") * 255
            cv.rectangle(mask, (minX, minY), (maxX, maxY), (0, 0, 0), 1)
            cnts, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            grouped = Contour(cnts[1], frame.shape, isEqualSign=True)
            # mark grouped contours for removal and add new grouped contour
            bar.remove = True
            bar2.remove = True
            contourList.append(grouped)

        # sort fraction bars ascending by width
        fractionBars.sort(key=lambda bar: bar.width)

        # group contours to fractions starting with most narrow fraction bar
        for bar in fractionBars:

            # build fraction
            fraction = Fraction(bar, contourList, frame.shape)

            # build new contour
            groupedContour = Contour(
                fraction.get_contour(), frame.shape, fraction=fraction)

            groupedContours = [*fraction.nominator,
                               *fraction.denominator, fraction.bar]

            # mark all grouped contours for removal and add new contour to contourList
            for cnt in groupedContours:
                cnt.remove = True
            contourList.append(groupedContour)

        # remove contours which were marked for removal before
        contourList = [cnt for cnt in contourList if not cnt.remove]

        return contourList

        #cv.drawContours(
        #    frame, [cnt.contour for cnt in contourList], -1, (0, 255, 0), 2)
