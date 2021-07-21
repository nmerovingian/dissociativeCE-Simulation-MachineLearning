import numpy as np
import time
from coeff import Coeff
from grid import Grid
from Solver import LU_solver
import csv
import scipy
from scipy import sparse
from scipy.sparse import linalg
import sympy
import time
import os
import math
from mailTest import sendMail
dElectrode = 1e-6
DA = 1e-9



def simulation(variable1:float,variable2:float,variable3:float,directory:str='.')->None:
    # if the save directory not exist, make the directory
    if not os.path.exists(directory):
        os.mkdir(directory)

    # the total concentration of X added before any chemical equilibrium happen 
    cTstar = 1e-3
    

    #start and end potential of voltammetric scan
    theta_i =20.0
    theta_v = -20.0


    #space step
    deltaX = 1e-6
    #potential step
    deltaTheta = 5e-3
    #expanding grid factor
    gamma = 0.05



    # standard electrochemical rate constant. Only useful if using Butler-Volmer equation for boundary conditions
    K0 = 1e9
    alpha = 0.5

    dimScanRate = variable1
    sigma = dElectrode*dElectrode/DA*(96485/(8.314*298)) *dimScanRate

    # equilibirum constants 
    dimKeq = variable2
    
    # forward reaction rate constants
    dimKf = variable3
    # reverse reaction rate constants
    dimKb = dimKf/ dimKeq

    # convert dimensional ones to dimensionless rate constants
    Kf = dimKf * dElectrode*dElectrode/DA

    Kb = dimKb*cTstar*dElectrode*dElectrode / DA

    print(f'Kf is {Kf},kb is {Kb}')

    keq = dimKeq / cTstar 


    #Get the bulk concentration of A after equilibrium
    cAstar = cTstar-(-1.0 + np.sqrt(1.0+4.0*cTstar/dimKeq))/(2.0/dimKeq)
    print(cAstar)

    concA = float(cAstar / cTstar)
    concB = float(np.sqrt(cAstar*dimKeq)/cTstar)
    concZ = float(np.sqrt(cAstar*dimKeq) / cTstar)
    concY = 0.0

    print(f"concA {concA}, concB{concB}, concY {concY}, concZ {concZ}")

    # dimensionless diffusion coefficients of every species 
    dB = 1.0
    dY = 1.0
    dZ = 1.0

    # the maximum number of iterations for Newton method
    number_of_iteration = 10

    deltaT = deltaTheta/sigma
    # The maximum distance of simulation
    maxT = 2.0*abs(theta_v-theta_i)/sigma
    maxX = 6.0 * np.sqrt(maxT)


    Print = False # If true, will print the concentration profiles
    printA = True
    printB = True
    printC = True
    printD = True
    printE = True 

    if not Print:
        printA = False
        printB = False
        printC = False
        printD = False
        printE = False

    pointA = -5.0
    pointB = -4.0
    pointC = -3.0
    pointD = theta_i
    pointE = -2.0


    # create the csv file to save data
    CVLocation  = f'{directory}/var1={variable1:.10f}var2={variable2:.10f}var3={variable3:.10f}.csv'

    concALocation = f"concA={concA:.2f}var1={variable1:.10f}var2={variable2:.10f}var3={variable3:.10f}.csv"
    concBLocation = f"concB={concB:.2f}var1={variable1:.10f}var2={variable2:.10f}var3={variable3:.10f}.csv"
    concYLocation = f"concY={concY:.2f}var1={variable1:.10f}var2={variable2:.10f}var3={variable3:.10f}.csv"
    concZLocation = f"concZ={concZ:.2f}var1={variable1:.10f}var2={variable2:.10f}var3={variable3:.10f}.csv"


    coeff = Coeff(deltaT,maxX,K0,alpha,gamma,Kf,Kb,dB,dY,dZ,concB)
    coeff.calc_n(deltaX)

    #simulation steps
    m = int(2.0*np.fabs(theta_v-theta_i)/deltaTheta)

    print(m)
    # initialzie matrix for Coeff object
    coeff.ini_jacob()
    coeff.ini_fx()
    coeff.ini_dx()
    # initialze matrix for Grid objectd
    grid = Grid(coeff.n)
    grid.grid(deltaX,gamma)
    grid.init_c(concA,concB,concY,concZ,theta_i)

    coeff.get_XX(grid.x)
    coeff.update(grid.conc,concA,concB,concY,concZ)
    coeff.Allcalc_abc(deltaT,theta_i,deltaX)
    coeff.calc_jacob(grid.conc,theta_i)
    coeff.calc_fx(grid.conc,theta_i)

    #print(np.linalg.det(coeff.J),coeff.J.shape,np.linalg.matrix_rank(coeff.J))




    #_,inds = sympy.Matrix(coeff.J).T.rref()
    #print(inds)
    #coeff.dx = LU_solver(coeff.J,coeff.fx)
    # use spsolve for sparse matrix for acceleration
    coeff.dx = linalg.spsolve(sparse.csr_matrix(coeff.J),sparse.csr_matrix(coeff.fx[:,np.newaxis]))
    #print(coeff.dx)
    coeff.xupdate(grid.conc,theta_i)

    for i in range(number_of_iteration):
        coeff.calc_jacob(grid.conc,theta_i)
        coeff.calc_fx(grid.conc,theta_i)
        #coeff.dx = np.linalg.solve(coeff.J,coeff.fx)
        #coeff.dx = scipy.linalg.solve_banded((4,4),coeff.J,coeff.fx)
        coeff.dx = linalg.spsolve(sparse.csr_matrix(coeff.J),sparse.csr_matrix(coeff.fx[:,np.newaxis]))
        grid.conc = coeff.xupdate(grid.conc,theta_i)
        if np.mean(np.absolute(coeff.dx)) < 1e-12:
            print('Exit: Precision satisfied!')
            break

    if printD and math.isclose(pointD,theta_i,rel_tol=1e-3):
        grid.updateAll()
        s = f'{directory}/Point=D,Theta={theta_i}{concALocation}'
        grid.saveA(s)
        s = f'{directory}/Point=D,Theta={theta_i}{concBLocation}'
        grid.saveB(s)
        s = f'{directory}/Point=D,Theta={theta_i}{concYLocation}'
        grid.saveY(s)
        s = f'{directory}/Point=D,Theta={theta_i}{concZLocation}'
        grid.saveZ(s)

        print('Saving point D')

        printD = False

    

    f=open(CVLocation,mode='w',newline='')

    writer = csv.writer(f)

    writer.writerow([theta_i,grid.grad()])
    Theta = theta_i
    #Theta = -7.0

    start_time = time.time()

    for i in range(m):
        
        if i < (m/2):
            Theta  = Theta - deltaTheta

        else:
            Theta = Theta + deltaTheta

        #Theta  = Theta + deltaTheta
        
        if i == 2:
            print(f'Total run time is {(time.time()-start_time)*m/60:.2f} mins')

        #grid.init_c(concA,concB,concY,concZ,Theta)

        coeff.update(grid.conc,concA,concB,concY,concZ)
        coeff.Allcalc_abc(deltaT,Theta,deltaX)
        for ii in range(number_of_iteration):
            coeff.calc_jacob(grid.conc,Theta)
            coeff.calc_fx(grid.conc,Theta)
            try:
                #coeff.dx = np.linalg.solve(coeff.J,coeff.fx)
                #coeff.dx = scipy.linalg.solve_banded((4,4),coeff.J,coeff.fx)
                coeff.dx=linalg.spsolve(sparse.csr_matrix(coeff.J),sparse.csr_matrix(coeff.fx[:,np.newaxis]))
            except:
                print("Using lstsq solver! ")
                coeff.dx = np.linalg.lstsq(coeff.J,coeff.fx,rcond=None)[0]
            grid.conc = coeff.xupdate(grid.conc,Theta)

            if np.mean(np.absolute(coeff.dx)) < 1e-12:
                #print(f'Exit: Precision satisfied!\nExit at iteration {ii}')
                break
            
        if not np.isnan(grid.grad()):
            writer.writerow([Theta,grid.grad()])
        else:
            print('Bad solution')

        #Save the concentration profile

        if printA and math.isclose(pointA,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=A,Theta={Theta}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=A,Theta={Theta}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=A,Theta={Theta}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=A,Theta={Theta}{concZLocation}'
            grid.saveZ(s)

            print('Saving point A')

            printA = False
        
        if printB and math.isclose(pointB,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=B,Theta={Theta}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=B,Theta={Theta}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=B,Theta={Theta}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=B,Theta={Theta}{concZLocation}'
            grid.saveZ(s)


            print('Saving point B')

            printB = False

        if printC and math.isclose(pointC,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=C,Theta={Theta}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=C,Theta={Theta}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=C,Theta={Theta}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=C,Theta={Theta}{concZLocation}'
            grid.saveZ(s)

            print('Saving point C')

            printC = False

        if printE and math.isclose(pointE,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=E,Theta={Theta}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=E,Theta={Theta}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=E,Theta={Theta}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=E,Theta={Theta}{concZLocation}'
            grid.saveZ(s)

            print('Saving point E')

            printE = False

    f.close()




    massConservation = grid.massConservation(concA,concB,concY,concZ)


    with  open(f'{directory}/massConservation.txt',mode='a') as f:
        f.write(f'{variable3},{massConservation}\n')
        


    


    













if __name__ == "__main__":
    simulation(1e0,1e-3,1e4,'./Conc Profile')

    

    


