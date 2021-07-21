import numpy as np
import csv

class Grid(object):
    def __init__(self,n):
        self.n = n
        self.x = np.zeros(self.n,dtype=np.float64)
        self.conc = np.zeros(self.n*4,dtype=np.float64)
        self.concA = np.zeros(self.n,dtype=np.float64)
        self.concB = np.zeros(self.n,dtype=np.float64)
        self.concY = np.zeros(self.n,dtype=np.float64)
        self.concZ = np.zeros(self.n,dtype=np.float64)

        self.g = 0.0


    def grid(self,dX,gamma):
        self.x[0] = 1.0
        for i in range(1,self.n):
            self.x[i] = self.x[i-1] + dX
            dX = dX* (1.0 +gamma )

    
    # initialize the concentration matrix
    def init_c(self,A:float,B:float,Y:float,Z:float,Theta:float):
        self.conc[::4] = A
        self.conc[1::4] = B
        self.conc[2::4] = Y
        self.conc[3::4] = Z

        self.concA[:] = A
        self.concB[:] = B
        self.concY[:] = Y 
        self.concZ[:] = Z

    # A better initialized concentration vector
    
    
    
    """
    def init_c(self,A:float,B:float,Y:float,Z:float,Theta:float):
        NernstB = B*1.0/(1.0 + np.exp(-Theta))
        NernstY = B*1.0/(1.0 + np.exp(Theta))

        self.concB = np.linspace(NernstB,B,num=self.n,endpoint=True)
        self.concY = np.linspace(NernstY,Y,num=self.n,endpoint=True)
        self.concZ[:] = Z
        self.concA[:] = A

        self.conc[::4] = self.concA
        self.conc[1::4] = self.concB
        self.conc[2::4] = self.concY
        self.conc[3::4] = self.concZ

        #print(self.conc)
    """
    """
    
    def init_c(self,A:float,B:float,Y:float,Z:float,Theta:float):
        NernstB = B*1.0/(1.0 + np.exp(-Theta))
        NernstY = B*1.0/(1.0 + np.exp(Theta))

        self.conc[::4] = A
        self.conc[1::4] = B
        self.conc[2::4] = Y
        self.conc[3::4] = Z

        self.conc[1] = NernstB
        self.conc[2] = NernstY

        self.concA = self.conc[::4]
        self.concB = self.conc[1::4]
        self.concY = self.conc[2::4]
        self.concZ = self.conc[3::4]  

    """
    def grad(self):
        self.g = -(self.conc[5]-self.conc[1]) / (self.x[1]-self.x[0])

        return self.g

    def updateAll(self):
        self.concA = self.conc[::4]
        self.concB = self.conc[1::4]
        self.concY = self.conc[2::4]
        self.concZ = self.conc[3::4]

    def saveA(self,filename):
        f=open(filename,mode='w',newline='')
        writer = csv.writer(f)
        for i in range(self.n):
            writer.writerow([self.x[i],self.concA[i]])
        f.close()

    def saveB(self,filename):
        f=open(filename,mode='w',newline='')
        writer = csv.writer(f)
        for i in range(self.n):
            writer.writerow([self.x[i],self.concB[i]])
        f.close()
    
    def saveY(self,filename):
        f=open(filename,mode='w',newline='')
        writer = csv.writer(f)
        for i in range(self.n):
            writer.writerow([self.x[i],self.concY[i]])
        f.close()
    
    def saveZ(self,filename):
        f=open(filename,mode='w',newline='')
        writer = csv.writer(f)
        for i in range(self.n):
            writer.writerow([self.x[i],self.concZ[i]])
        f.close()


    def massConservation(self,A,B,Y,Z):
        massInitial = 0.0
        massA = 0.0
        massB = 0.0
        massY = 0.0
        massZ = 0.0 
        tempA = 0.0
        tempB = 0.0
        tempY = 0.0
        tempZ = 0.0
        for i in range(self.n-1):
            h = self.x[i+1] - self.x[i]
            massInitial += h*(2*A+B+Y+Z)
            tempA = (self.concA[i] + self.concA[i + 1]) / 2.0
            massA += 2.0*tempA * h
            tempB = (self.concB[i] + self.concB[i + 1]) / 2.0
            massB += tempB * h
            tempY = (self.concY[i] + self.concY[i + 1]) / 2.0
            massY += tempY * h
            tempZ = (self.concZ[i] + self.concZ[i + 1]) / 2.0
            massZ += tempZ * h

        massFinal = massA + massB + massY + massZ

        return (massFinal-massInitial)/massInitial