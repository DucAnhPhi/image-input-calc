import numpy as np
import cv2

from preprocessing import PreProcessing


def ReturnN(t):
    return t.n

class SubImage:

    def __init__(self, PointVector,InIm):

        boundary = 10
        n = PointVector.shape[0]
        if(n==0):
            print("Fatal error: An empty PointVector was submitted")
            cv2.waitKey(0)
            Image = np.zeros(shape=(0,0))
            x = 0
            y = 0
            xsize = 0
            ysize = 0
            rot = 0
        else:
            #print("New: ###############################")
            #print("n: ", n)

            xmin = np.Infinity
            xmax = 0

            ymin = np.Infinity
            ymax = 0

            for i in range(n):
                if (PointVector[i][1] < xmin):
                    xmin = PointVector[i][1]
                if (PointVector[i][1] > xmax):
                    xmax = PointVector[i][1]
                if (PointVector[i][0] < ymin):
                    ymin = PointVector[i][0]
                if (PointVector[i][0] > ymax):
                    ymax = PointVector[i][0]



            #xmin = xmin - boundary
            #xmax = xmax + boundary
            #ymin = ymin - boundary
            #ymax = ymax + boundary

            self.rot = 0

            self.x = (int)(xmax + xmin)/2
            self.y = (int)(ymax + ymin)/2

            self.xsize = (int)(xmax - xmin) + 1
            self.ysize = (int)(ymax - ymin) + 1


            #self.Image = InIm[ymin:ymax, xmin:xmax]  # ZSim
            #cv2.imshow('Subimage', self.Image)
            #print("Range: ( (", ymin, ",", ymax, "),(", xmin, ",", xmax, ")")


            #print(PointVector)
            #PointVector[:][0] = PointVector[:][0] - ymin +boundary
            #PointVector[:][1] = PointVector[:][1] - xmin + boundary

            ZSim = np.zeros(shape=[self.ysize, self.xsize, 1])  # , dtype=np.uint8)

            #print("Size: (",self.ysize,",",self.xsize,")")
            #print("n = ", n)
            #print(PointVector)
            for i in range(n):
                #print("(",(PointVector[i][0]-ymin),",",(PointVector[i][1]-xmin),")" )
                ZSim[(int)(PointVector[i][0] - ymin)][(int)(PointVector[i][1] - xmin)]=255

                #ZSim[(int)(PointVector[i][0])][(int)(PointVector[i][1])]

            self.Image=ZSim
            self.n=n


    Image = None
    x = 0
    y = 0
    xsize = 0
    ysize = 0
    rot = 0
    n=0


def ConnectedCutting(InIm):
    #We don't want to change the original image
    ZS=InIm.copy()
    #We want to return this
    SubDivList = []


    h = InIm.shape[0]
    w = InIm.shape[1]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    ZSVectorList = []


    #cv2.imshow('Before', ZS)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    BlackPix = np.argwhere(ZS == 0)
    print("Found ", BlackPix.shape[0], " black pixels")

    #print("Point 0")
    while BlackPix.shape[0]>0:
        #print((BlackPix[0][0], BlackPix[0][1]))
        cv2.floodFill(ZS, mask, (BlackPix[0][1], BlackPix[0][0]), 100);

        #print(BlackPix.shape[0], " Black Pixels left")
        #cv2.waitKey(0)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        ZSVectorList = np.argwhere(ZS == 100)

        cv2.floodFill(ZS, mask, (BlackPix[0][1], BlackPix[0][0]), 255);



        if(len(ZSVectorList)>0):
            ZSSubIm = SubImage(ZSVectorList,InIm)
            #print(ZSSubIm.Image)
            if (ZSSubIm == None):
                print("Fatal Error")
            #cv2.imshow("SubImage :", ZSSubIm.Image)
            #cv2.waitKey(0)
            SubDivList.append(ZSSubIm)

        else:
            print("How?")
        BlackPix = np.argwhere(ZS == 0)


        #print("Found ", BlackPix.shape[0], " black pixels")

        #Usefull for Debugging
        #print(ZSVectorList)
        #print(ZSSubIm)
        #cv2.imshow('ZSSubIm: ', ZSSubIm.Image)
        #cv2.imshow('ZS: ', ZS)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    #print("Point 4")



    SubDivList.sort(key=ReturnN)
    return SubDivList
    # source: https://www.programcreek.com/python/example/89425/cv2.floodFill

def PolylineFitting(InIm):
    PLIm=InIm.copy()
    PLIm = cv2.resize(InIm, None, fx=0.5, fy=0.5)
    contours, hierarchy = cv2.findContours(PLIm,1,2)

    print("We found ", len(contours), " contours")

    PLIm = cv2.cvtColor(PLIm, cv2.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        cnt = contours[i]
        # print(cnt)
        M = cv2.moments(cnt)
        # print(M)
        # cv2.waitKey(0)
        if (int(M['m00']) == 0):
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt, True)

        epsilon = 0.1 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # PYTHON SUCKS


        approx.reshape((-1, 1, 2))
        # print("approx1:")
        # print(approx.shape)
        approx = approx[:, 0, :]
        # print("approx:")
        # print(approx)
        cv2.polylines(PLIm, [approx], True, (0, 0, 255))
        pts = np.array([[1, 1], [539, 539], [539, 0]])
        pts.reshape((-1, 1, 2))
        # print("pts")
        # print(pts)
        # cv2.polylines(PLIm, [pts], True, (0))
        # for i in range(approx.shape[0]):
        #    PLIm[approx[i][0][1]][approx[i][0][0]]=50
        # print(InIm.shape)

        # print(approx)
    cv2.imshow("Approximated:", PLIm)
    cv2.waitKey(0)












































    #
    # Failed attempts. Simply ignore. Please don't delete, I use it for reference
    #



    '''def BorderRemoval(InIm):
    #PLIm = cv2.resize(InIm, None, fx=0.5, fy=0.5)
    PLIm=InIm.copy()
    PLIm = cv2.copyMakeBorder(PLIm.copy(), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0,0,0))
    PLCopy = PLIm.copy()
    kernel=np.ones((5,5))
    for i in range(5):
        #cv2.morphologyEx(PLIm, cv2.MORPH_TOPHAT, kernel, iterations=1)
        PLIm=cv2.erode(PLIm,kernel)
        #PLIm = cv2.GaussianBlur(PLIm, (9, 9), 0)
    PLIm=PreProcessing().mediumbinarize(PLIm)

    h = PLIm.shape[0]
    w = PLIm.shape[1]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(PLIm, mask, (1,1), 255);

    PLIm=np.where(PLIm==0, PLCopy, 255)

    PLIm=PLIm[10:h-10,10:w-10]

    return PLIm'''

    '''#contours, hierarchy = cv2.findContours(PLIm,1,2)

    #print("We found ", len(contours), " contours")

    PLIm = cv2.cvtColor(PLIm, cv2.COLOR_GRAY2RGB)
    for i in range(2):
        cnt = contours[i]
        #print(cnt)
        M = cv2.moments(cnt)
        #print(M)
        #cv2.waitKey(0)
        if(int(M['m00'])==0):
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt,True)

        epsilon=0.05 * perimeter
        approx= cv2.approxPolyDP(cnt,epsilon,True)


        #PYTHON SUCKS


        approx.reshape((-1, 1, 2))
        #print("approx1:")
        #print(approx.shape)
        approx=approx[:,0,:]
        #print("approx:")
        #print(approx)
        cv2.polylines(PLIm,[approx],True,(0,0,255))
        pts = np.array([[1,1],[539,539],[539,0]])
        pts.reshape((-1, 1, 2))
    #print("pts")
    #print(pts)
    #cv2.polylines(PLIm, [pts], True, (0))
    #for i in range(approx.shape[0]):
    #    PLIm[approx[i][0][1]][approx[i][0][0]]=50
    #print(InIm.shape)
    #cv2.imshow("Approximated:", PLIm)
    #cv2.waitKey(0)
    #print(approx)'''