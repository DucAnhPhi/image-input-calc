import cv2 as cv
import numpy as np


from segmentation import Segmentation
from drawing import Draw



class LineOrdering2:

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

    def get_medianVector(self, vectorList):
        if (len(vectorList) == 0):
            return (0, 0)
        xList=[]
        yList=[]
        for i in range(len(vectorList)):
            xList.append(vectorList[i][0])
            yList.append(vectorList[i][1])
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

    # This returns a List of contours that are larger than the cutOffRadius
    # Function Not Used
    def get_contoursLargerThanRadius(self,contours, cutOffRadius, rList=None):
        if rList==None:
            xList, yList, rList = Segmentation().get_properties_mincircle(contours)
        contoursLargerThanRadius=[]
        for i in range(len(contours)):
            if rList[i] >cutOffRadius:
                contoursLargerThanRadius.append(contours[i])

        return contoursLargerThanRadius

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
    def horVec_sorter(self, contours, horVec):

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

    #sort contours by size
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
    def get_largestVector(self, contours,fullVectorList):
        xList, yList, rList = Segmentation().get_properties_mincircle(contours)

        #get coordinate tupels (y[i],x[i]) instead of separate lists
        p = self.get_coords_list(yList,xList)



        maxLength=0
        maxLengthIndex=-1
        maxVector=(0.1,0)

        for i in range(len(fullVectorList)):
            if np.linalg.norm(fullVectorList[i])>maxLength:
                maxLength=np.linalg.norm(fullVectorList[i])
                maxLengthIndex=i
                maxVector=fullVectorList[maxLengthIndex]


        return maxVector



    # Here we determine the direction of the line of text.
    def get_horVec2(self, contours):
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
        xList, yList, rList = Segmentation().get_properties_mincircle(contours)
        points = self.get_coords_list(yList, xList)

        # we calculate here a radius that should be ideally larger than the points and smaller than the symbols.
        cutOffRadius=np.mean(rList)/2

        # We get the full Vector list here (and the one of Vectors between contours larger than cutOffRadius, which is actually a lot more usefull)
        fullVectorList,largeContourVectorList=self.get_fullVectorList_and_largeContourVectorList(rList,points,cutOffRadius)

        # We directionalise with the help of largest Vector. This Vector is not likely parallel to the horVec.
        # However, it will likely not be close to orthogonal to it, which is important in this case.
        directionalisedVectorList=self.directionalize(fullVectorList,self.get_largestVector(contours,fullVectorList))#fullVectorList))

        # We normalise the Vectors, because far of contours produce large Vectors. By normalising all Vectors, we reduce this problem.
        # The process also works without. But this is an improvement
        print("B: We found ",len(directionalisedVectorList), " vectors")
        for i in range(len(directionalisedVectorList)):
            directionalisedVectorList[i]=self.normalise_vector(directionalisedVectorList[i])

        # We calculate the mean Vector
        horVec=self.get_medianVector(directionalisedVectorList)


        #to avoid it getting integerized out of existance
        horVec=self.scalarVectorMul(horVec,100)

        print("horVec = ", horVec)

        return horVec








    # get a list of the position of all lines in orthogonal Distance.
    def get_linePositionList(self,contours, orthDistList,maxRad):
        minOrthDist=np.min(orthDistList)
        maxOrthDist=np.max(orthDistList)

        linePositionList=[]

        currentPoint=minOrthDist

        while currentPoint < maxOrthDist:
            linePositionList.append(currentPoint)
            currentPoint=currentPoint+maxRad*0.4

        return linePositionList,0



    # get the ordered line lists of the contours. The 2 is here because this is the second attempt at this.
    def get_orderedLineList2(self,contours,inIm):
        # check if contours were detected. If not, nothing can be done here, except for errors produced
        if len(contours)==0:
            print("ERROR NO CONTOURS DETECTED")
            cv.waitKey()
            return None, (0,0), inIm

        #determining some variables to be used later
        xList, yList, rList = Segmentation().get_properties_mincircle(contours)
        maxRad = max(rList)
        cutOffRadius = np.mean(rList)
        lineAcceptanceRadius=0.5*maxRad

        # get the direction in which the line is written
        horVec= self.get_horVec2(contours)

        # determine the distance from (0,0) have along the orthogonal axis to the writing direction
        orthDistList=self.get_OrthDistList(contours, horVec)

        # determine the position of all the lines in the image
        linePositionList,nearestContourIndex= self.get_linePositionList(contours,orthDistList,maxRad)

        # Just feedback
        inIm = Draw().LineFeedBack(inIm,contours,contours[nearestContourIndex],orthDistList,horVec,maxRad)



        # We will return this. Time to fill it
        orderedLineList = []
        # We add the lines separately
        toAddLine = []

        # print("maxRad = ", maxRad)
        # print("Area of Acceptance: ", str(linePositionList[0] + lineAcceptanceRadius) , " to ", str(linePositionList[0] - lineAcceptanceRadius))

        # In case you want to change it
        contoursUsed = contours.copy()

        # For every line we check every contour if it fits.
        for i in range(len(linePositionList)):
            toAddLine=[]
            for j in range(len(contoursUsed)):
                if orthDistList[j]< (linePositionList[i] + lineAcceptanceRadius) and orthDistList[j] > (linePositionList[i] - lineAcceptanceRadius):
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


