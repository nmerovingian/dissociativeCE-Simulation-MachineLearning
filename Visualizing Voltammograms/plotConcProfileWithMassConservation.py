from helper import find_csv,find_sigma,find_conc,find_point
from OxTafelSlope import get_apparent_transfer_coefficient
import pandas as pd 
import matplotlib.pyplot as plt
import os
import re
import numpy as np 








def plot_points(cvs):
    fig, ax = plt.subplots(figsize=(16,9))

    point,Theta, sigma, Kf, Kb =  find_conc(cvs[0])

    for cv in cvs:
        df = pd.read_csv(cv,header=None)
        species = 'Unknown'
        if 'concA'  in cv:
            species = 'X'
            #ax.set_title(f'Concentration Profile\nPoint {point}, $\\theta={Theta:.2f}$\n $d\Gamma={sigma}$ Flux = {(-(df.iloc[1,1]-df.iloc[2,1])/Kf):.2f}')
            ax.set_title(f'Concentration Profile\nPoint {point}, $\\theta={Theta:.2f}$\n $d\\theta={sigma:.2f}$,$k_{{eq}}={Kf:.2E}M$,$k_{{f}}={Kb:.2E}s^{{-1}}$')
        elif 'concB'  in cv:
            species = 'A'
        elif 'concY'  in cv:
            species = 'B'
        elif 'concZ' in cv:
            species = 'C'

        df = df.iloc[0:21]
        #df[0] = 1.0/(1.0-df[0])
        if species == 'A':
            df.plot(x=0,y=1,marker='x',ax=ax,label=f'{species}',linewidth=linewidth,alpha=0.5,markersize=20)
        else:
            df.plot(x=0,y=1,marker='o',ax=ax,label=f'{species}',linewidth=linewidth,alpha=0.5)


    #Consider mass conservation , at the same point of x A + 2B + 3C =1 
    

    for cv in cvs:
        if 'concA' in cv:
            total = pd.read_csv(cv,header=None)

    for cv in cvs:
        df = pd.read_csv(cv,header=None)
        if 'concA' in cv:
            total[1] = 2*total[1]
        elif 'concB' in cv:
            total[1] += df[1]
        elif 'concY' in cv:
            total[1] += df[1]
        elif "concZ" in cv:
            total[1] += df[1]

    total = total.iloc[0:21]
    #total[0] = 1.0/(1.0-total[0])
    total.plot(x=0,y=1,marker='o',ax=ax,label='2X+A+B+C',linewidth=linewidth)



    #ax.set_xlabel(r'$r/r_{{sphere}}$')
    ax.set_xlabel(r'$\Gamma$')
    ax.set_ylabel('Concentration')
    
    fig.savefig(f'Point{point}.png',dpi=400)

    


if __name__ == "__main__":
    
    CVs = find_csv()

    linewidth = 3
    fontsize = 20
    figsize = [10,8]

    font = {'family' : 'monospace',
            'weight' : 'bold',
            'size'   : fontsize }
    plt.rc('font', **font)  # pass in the font dict as kwargs


    collection = ['D']

    for point in collection:
        points = find_point(CVs,point)
        plot_points(points)


