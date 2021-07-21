import numpy as np

class Coeff(object):
    def __init__(self,deltaT,maxX,K0,alpha,gamma,Kf,Kb,dB,dY,dZ,concB):
        self.n = 0
        self.xi = 0.0
        self.maxX = maxX
        self.K0 = K0
        self.alpha = alpha
        self.gamma = gamma
        self.Kf = Kf*deltaT
        self.Kb = Kb*deltaT
        self.dB = dB
        self.dY = dY
        self.dZ = dZ
        self.concB = concB



    def calc_n(self,dX):
        while self.xi < self.maxX:
            self.xi += dX
            dX = dX*(1.0+self.gamma)
            self.n = self.n + 1

        self.n = self.n + 1 

        print(f"n is {self.n}")

        self.aA = np.zeros(self.n)
        self.bA = np.zeros(self.n)
        self.cA = np.zeros(self.n)
        self.aB = np.zeros(self.n)
        self.bB = np.zeros(self.n)
        self.cB = np.zeros(self.n)
        self.aY = np.zeros(self.n)
        self.bY = np.zeros(self.n)
        self.cY = np.zeros(self.n)
        self.aZ = np.zeros(self.n)
        self.bZ = np.zeros(self.n)
        self.cZ = np.zeros(self.n)
        self.d = np.zeros(self.n*4)

        self.XX = np.zeros(self.n)
    

    def ini_jacob(self):
        self.J = np.zeros((4*self.n,4*self.n),dtype=np.float64)


    def ini_fx(self):
        self.fx = np.zeros(4*self.n,dtype=np.float64)


    def ini_dx(self):
        self.dx = np.zeros(4*self.n,dtype=np.float64)

    def get_XX(self,xx:np.ndarray):
        self.XX = xx.copy()


    # D value are different for a steady state simulation
    def update(self,x,A,B,Y,Z):
        self.d = x.copy()

        self.d[-1] = Z
        self.d[-2] = Y
        self.d[-3] = B
        self.d[-4] = A

    
    def xupdate(self,x,Theta):
        x = x+ 1e-1*self.dx
        return x
    
    
    """
    def xupdate(self,x,Theta):
        t = 1.0

        self.calc_fx(x,Theta)
        original = np.mean(np.absolute(self.fx))
        self.calc_fx(x+t*self.dx,Theta)
        damped = np.mean(np.absolute(self.fx))

        while damped > original:
            t = t/2.0
            self.calc_fx(x+t*self.dx,Theta)
            damped = np.mean(np.absolute(self.fx))

            #print(t)

            if t < 1e-10:
                break

        x = x + t*self.dx
        return x"""
    
    
    

    def Acal_abc(self,deltaT,Theta,deltaX):
        self.aA[0] = 0.0
        self.bA[0] = 0.0
        self.cA[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.XX[i] - self.XX[i - 1]
            deltaX_p = self.XX[i + 1] - self.XX[i]
            self.aA[i] = (-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))
            self.bA[i] = ((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p))) + 1.0
            self.cA[i] = (-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))

        self.aA[-1] = 0.0
        self.bA[-1] = 0.0
        self.cA[-1] = 0.0

    def Bcal_abc(self,deltaT,Theta,deltaX):
        self.aB[0] = 0.0
        self.bB[0] = 0.0
        self.cB[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.XX[i] - self.XX[i - 1]
            deltaX_p = self.XX[i + 1] - self.XX[i]
            self.aB[i] = (-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))
            self.bB[i] = ((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p))) + 1.0
            self.cB[i] = (-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))

        self.aB[-1] = 0.0
        self.bB[-1] = 0.0
        self.cB[-1] = 0.0

    def Ycal_abc(self,deltaT,Theta,deltaX):
        self.aY[0] = 0.0
        self.bY[0] = 0.0
        self.cY[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.XX[i] - self.XX[i - 1]
            deltaX_p = self.XX[i + 1] - self.XX[i]
            self.aY[i] = (-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))
            self.bY[i] = ((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p))) + 1.0
            self.cY[i] = (-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))
        self.aY[-1] = 0.0
        self.bY[-1] = 0.0
        self.cY[-1] = 0.0

    def Zcal_abc(self,deltaT,Theta,deltaX):
        self.aZ[0] = 0.0
        self.bZ[0] = 0.0
        self.cZ[0] = 0.0

        for i in range(1,self.n-1):
            deltaX_m = self.XX[i] - self.XX[i - 1]
            deltaX_p = self.XX[i + 1] - self.XX[i]
            self.aZ[i] = (-(2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p)) + 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))
            self.bZ[i] = ((2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) + (2.0 * deltaT) / (deltaX_m * (deltaX_m + deltaX_p))) + 1.0
            self.cZ[i] = (-(2.0 * deltaT) / (deltaX_p * (deltaX_m + deltaX_p)) - 2.0 / self.XX[i] * (deltaT / (deltaX_m + deltaX_p)))
        self.aZ[-1] = 0.0
        self.bZ[-1] = 0.0
        self.cZ[-1] = 0.0

    def Allcalc_abc(self,deltaT,Theta,deltaX):
        self.Acal_abc(deltaT,Theta,deltaX)
        self.Bcal_abc(deltaT,Theta,deltaX)
        self.Ycal_abc(deltaT,Theta,deltaX)
        self.Zcal_abc(deltaT,Theta,deltaX)


    def calc_jacob(self,x,Theta):
        h = self.XX[1] - self.XX[0]

        self.J[0,0] = -1.0
        self.J[0,4] = 1.0
        """
        #Nernst Condition
        self.J[1,2] = -((self.dB*np.exp(Theta)+self.dY)/h)
        self.J[1,5] = self.dB / h
        self.J[1,6] = self.dY / h
        self.J[2,1] = -((self.dY/np.exp(Theta)+self.dB)/h)
        self.J[2,5] = self.dB / h
        self.J[2,6] = self.dY / h"""
        
        
        #BV boundary condition
        Kred = self.K0*np.exp(-self.alpha*Theta)
        Kox = self.K0*np.exp((1.0-self.alpha)*Theta)
    
        self.J[1,1] = (1.0+(1.0/self.dB)*Kred*h)
        self.J[1,2] = -(1.0/self.dB)*Kox*h
        self.J[1,5] = -1.0
        self.J[2,1] = (-1.0/self.dY)*Kred*h
        self.J[2,2] = ((1.0/self.dY)*Kox*h+1.0)
        self.J[2,6] =-1.0
        

        #self.J[2,5] = 1.0/ np.exp(Theta)
        #self.J[2,6] = -1.0

        #self.J[2,1] = -self.dB
        #self.J[2,2] = -self.dY
        #self.J[2,5] = self.dB
        #self.J[2,6] = self.dY
        
        
        #self.J[1,1] = 1.0

        #self.J[2,2] = -1.0

        
        self.J[3,3] = -1.0 
        self.J[3,7] = 1.0

        for row in range(4,self.n*4-4,4):
            i = int(row/4)

            #initialize species A:
            self.J[row,row-4] = self.aA[i]
            self.J[row,row] = self.bA[i] + self.Kf
            self.J[row,row+1] = -self.Kb * x[i+3]
            self.J[row,row+3] = -self.Kb*x[i+1]
            self.J[row,row+4] = self.cA[i]

            self.J[row+1,row-3] = self.aB[i]
            self.J[row+1,row] = -self.Kf
            self.J[row+1,row+1] = self.bB[i] + self.Kb*x[i+3]
            self.J[row+1,row+3] = self.Kb*x[i+1]
            self.J[row+1,row+5] = self.cB[i]

            self.J[row+2,row-2] = self.aY[i]
            self.J[row+2,row+2] = self.bY[i]
            self.J[row+2,row+6] = self.cY[i]

            self.J[row+3,row-1] = self.aZ[i]
            self.J[row+3,row] = -self.Kf
            self.J[row+3,row+1] = self.Kb*x[i+3]
            self.J[row+3,row+3] = self.bZ[i] + self.Kb*x[i+1]
            self.J[row+3,row+7] = self.cZ[i]

        self.J[4*self.n-4,4*self.n-4] = 1.0
        self.J[4*self.n-3,4*self.n-3] = 1.0
        self.J[4*self.n-2,4*self.n-2] = 1.0
        self.J[4*self.n-1,4*self.n-1] = 1.0


    def calc_fx(self,x,Theta):
        h = self.XX[1] - self.XX[0]
        self.fx[0] = x[4] - x[0]

        """
        # Nernst boundary condition 
        self.fx[1] = self.dB*((x[5]-x[2]*np.exp(Theta))/h) + self.dY*((x[6]-x[2])/h)
        self.fx[2] = self.dY*((x[6]-x[1]/np.exp(Theta))/h) + self.dB*((x[5]-x[1])/h)
        """
        
        #BV boundary condition
        Kred = self.K0*np.exp(-self.alpha*Theta)
        Kox = self.K0*np.exp((1.0-self.alpha)*Theta)
        self.fx[1] = (1.0 + (1.0 / self.dB) * Kred * h) * x[1] - (1.0 / self.dB) * Kox * h * x[2] - x[5]
        self.fx[2] = (-1.0 / self.dY ) * Kred * h * x[1] + ((1.0 / self.dY) * Kox * h + 1.0) * x[2] - x[6]
        

        #self.fx[2] = self.dY*((x[6]-x[2])/h) + self.dB*((x[5]-x[1])/h)
        #self.fx[2] = x[1]/np.exp(Theta) - x[2]

        #self.fx[1]  =  x[1] - self.concB * (1.0/(1.0+np.exp(-Theta)))
        #self.fx[2]  = self.concB* (1.0/(1.0+np.exp(Theta))) - x[2]
        self.fx[3] = x[7] - x[3]


        for j in range(4,4*self.n-4,4):
            i = int(j/4)

            self.fx[j] = self.aA[i]*x[4*i-4] + self.bA[i]*x[4*i]+ self.cA[i]*x[4*i+4] + self.Kf*x[4*i] -self.Kb*x[4*i+1]*x[4*i+3] - self.d[4*i]
            self.fx[j+1] = self.aB[i]*x[4*i-3] + self.bB[i]*x[4*i+1] + self.cB[i]*x[4*i+5] - self.Kf*x[4*i] + self.Kb*x[4*i+1]*x[4*i+3] - self.d[4*i+1]
            self.fx[j+2] = self.aY[i]*x[4*i-2] + self.bY[i]*x[4*i+2] + self.cY[i]*x[4*i+6] - self.d[4*i+2]
            self.fx[j+3] = self.aZ[i]*x[4*i-1] + self.bZ[i]*x[4*i+3] + self.cZ[i]*x[4*i+7] - self.Kf*x[4*i] + self.Kb*x[4*i+1]*x[4*i+3] - self.d[4*i+3]


        self.fx[4*self.n-4] = x[self.n*4-4] - self.d[4*self.n-4]
        self.fx[4*self.n-3] = x[self.n*4-3] - self.d[4*self.n-3]
        self.fx[4*self.n-2] = x[self.n*4-2] - self.d[4*self.n-2]
        self.fx[4*self.n-1] = x[self.n*4-1] - self.d[4*self.n-1]

        self.fx = -self.fx