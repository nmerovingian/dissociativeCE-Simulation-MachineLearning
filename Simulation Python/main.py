from concurrent.futures import ProcessPoolExecutor
import os
from simulation import simulation
import numpy as np 

def foo(a0,a1,a2):
    #print(a0,a1,a2)
    print('executed')
     



if __name__ == "__main__":

    variables1 = [1e-3] # SCAN RATE
    variables2 =  [1e-4]#keq
    variables3 =  [1e3]#kf
    directory  = ['./Data']

    total_elements = len(variables1) * len(variables2) * len(variables3)

    variables1 = list(variables1) * int((total_elements /len(variables1)))
    variables2 = list(variables2) * int(total_elements /len(variables2)) 
    variables3 = list(variables3) * int(total_elements /len(variables3))

    print(len(variables1),len(variables2),len(variables3))
    directory = directory * total_elements
    print(total_elements)
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(simulation,variables1,variables2,variables3,directory)



