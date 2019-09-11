import numpy as np
import cv2

from preprocessing import PreProcessing

def ReturnN(t):
    return t.n

def Returnx(t):
    return t.x

def Resize(SubImList, Pixellength=75):
    SubImListZS=SubImList.copy()
    xsize = []
    ysize = []
    max=[]





    SubImListZS.sort(key=ReturnN)
    mean=sum(SubImZS.n for SubImZS in SubImListZS)/len(SubImListZS)
    print("SubImList n mean: ", sum(SubImZS.n for SubImZS in SubImListZS)/len(SubImListZS))

    i=0
    while i <(len(SubImListZS)):
        if SubImListZS[i].n>5*mean:
            #print("Deleting element ", i , " : " , SubImListZS[i].n)
            del(SubImListZS[i])
        else:
            pass#SubImListZS.append(SubImList[i])
        i=i+1

    #print("It has ", len(SubImListZS) , " entries")
    for i in range(len(SubImListZS)):
        max.append(0)
        if (SubImListZS[i].xsize > max[i]):
            max[i] = SubImListZS[i].xsize
        if (SubImListZS[i].ysize > max[i]):
            max[i] = SubImListZS[i].ysize

    print("Highest weight/mean: ", SubImListZS[len(SubImListZS)-1].n/mean)
    #print("SubImList n: ")
    #for i in range(len(SubImListZS)):
    #    print(SubImListZS[i].n)

    mean = sum(SubImZS.n for SubImZS in SubImListZS) / len(SubImListZS)

    for i in range(len(SubImListZS)):

        #print("Weight of element ", i ," : ", SubImListZS[i].n)
        #print("Before:",SubImList[i].Image.shape[0],", ",SubImList[i].Image.shape[1])

        if(SubImListZS[i].n>mean):
            dim=(int(SubImListZS[i].Image.shape[1]*(Pixellength-2)/max[i]),int(SubImListZS[i].Image.shape[0]*(Pixellength-2)/max[i]))
            SubImZS=cv2.resize(SubImListZS[i].Image,dim,cv2.INTER_AREA)

            #print("After ",SubImZS.shape[0],", ",SubImZS.shape[1])

            background=np.zeros(shape=(Pixellength,Pixellength))
            for y in range(int(SubImZS.shape[0])):
                for x in range(int(SubImZS.shape[1])):
                    background[int(Pixellength/2-SubImZS.shape[0]/2)+y][int(Pixellength/2-SubImZS.shape[1]/2)+x]=SubImZS[y][x]
        else:

            SubImZS = SubImListZS[i].Image
            print(SubImZS.shape, " Pixellength: ", Pixellength)
            background = np.zeros(shape=(Pixellength, Pixellength))

            for y in range(np.minimum(SubImZS.shape[0],Pixellength-1)):
                for x in range(np.minimum(SubImZS.shape[1], Pixellength-1)):
                    #print("Test:", x, " ", y)
                    background[y][x] = SubImZS[y][x]
        SubImListZS[i].Image=background.copy()

    SymbolList = []
    PointList = []


    for i in range(len(SubImListZS)):
        if SubImListZS[i].n>mean:
            SymbolList.append(SubImListZS[i])
        else:
            PointList.append(SubImListZS[i])
    return SymbolList, PointList


def Orient(SymbolList, PointList,InIm):

    Radius=0
    ZSIm=InIm.copy()
    SymbolList.sort(key=Returnx)

    ZSIm = cv2.cvtColor(ZSIm, cv2.COLOR_GRAY2RGB)
    SubImList=SymbolList.copy()
    SubImPointList=PointList.copy()
    x=[]
    y=[]


    n=[]
    rad=[]
    for i in range(len(SubImList)):
        x.append(int(SubImList[i].x))
        y.append(int(SubImList[i].y))
        #print("X: ",x[i],"Y: ",y[i])
        rad.append(np.sqrt(SubImList[i].xsize*SubImList[i].xsize+SubImList[i].ysize*SubImList[i].ysize))

    Radius=1.2*np.mean(rad)

    print("")
    print("")
    print("")

    print("Radius of Acceptance: ", Radius)
    print("")


    LineList=[]
    while len(SubImList)>0:
        PointIndex=[]
        print("")
        print("Starting new Line.")
        print("Currently ", len(SubImList), " unassigned symbols left")
        LineZS = []
        LineZS.append(SubImList[0])
        del SubImList[0]
        changed=True
        while(changed):
            j=0
            i=0
            k=0
            changed=False
            while j <(len(LineZS)):
                if j in PointIndex:
                    pass
                else:
                    while k < (len(SubImPointList)):
                        d = np.sqrt((SubImPointList[k].x - LineZS[j].x) ** 2 + (SubImPointList[k].y - LineZS[j].y) ** 2)
                        if (d < 0.5*Radius):
                            print("Point number ", k, " is part of this line")
                            LineZS.append(SubImPointList[k])
                            del SubImPointList[k]
                            changed = True
                            PointIndex.append(len(LineZS)-1)


                            cv2.line(ZSIm, (int(LineZS[len(LineZS)-2].x), int(LineZS[len(LineZS)-2].y)),(int(LineZS[len(LineZS) - 1].x), int(LineZS[len(LineZS) - 1].y)), (0, 255, 0), 5)
                        k=k+1

                    while i <(len(SubImList)):
                        #print("Currently at i=",i,"/",len(SubImList),", j=",j,"/",len(LineZS))
                        d=np.sqrt((SubImList[i].x-LineZS[j].x)**2+(SubImList[i].y-LineZS[j].y)**2)
                        if(d<Radius):
                            print("Subimage number ", i, " is part of this line")
                            LineZS.append(SubImList[i])
                            del SubImList[i]
                            changed=True
                        i = i + 1
                        cv2.line(ZSIm, (int(LineZS[0].x),int(LineZS[0].y)),(int(LineZS[len(LineZS)-1].x),int(LineZS[len(LineZS)-1].y)), (0, 0, 255), 5)

                j=j+1
                i=0
                k=0



        LineList.append(LineZS)
    print("")
    print("")
    print("We have ", len(LineList), " lines")
    for i in range(len(LineList)):
        LineList[i].sort(key=Returnx)
        #for j in range(len(LineList[i])):







    #for i in range(len(SubImList)):
    #    for j in range(int(-Radius/2),int(Radius/2)):
    #        for k in range(int(-Radius/2),int(Radius/2)):
    #            ZSIm[y[i]+j][x[i]+k] = (0, 0, 255)



    #print(x,y,n,weight)
    #fit=np.polyfit(y,x,1)



    return LineList,ZSIm







    '''dist=np.zeros(shape=(len(SubImList),len(SubImList)))
    for i in range(len(SubImList)):
        #print(p[i])
        for j in range(len(SubImList)):
            if(i!=j):
                for k in range(len(SubImList)):
                    pass
    mean= np.mean(n)
    stddev = np.sqrt(np.var(n))
    '''
    #print(n)
    #print("Mean: ",mean, " stddev: ",stddev)
    #cv2.waitKey(0)
