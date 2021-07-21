from helper import find_csv,find_sigma,find_experimental_csv
from OxTafelSlope import get_apparent_transfer_coefficient
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import re
import numpy as np 
CVs = find_csv()
#CVs = CVs[::-1]
ExperimentalCVs = find_experimental_csv()

from matplotlib import cycler 
#plt.rcParams['axes.prop_cycle'] = cycler(linestyle=['-','-.','--',':','-','-.','--',':'],color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
colors = cm.viridis(np.linspace(0,1,len(CVs)))
index = 0

linewidth = 3
fontsize = 14
figsize = [10,8]

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


#Plot all OXCV together

fig = plt.figure(figsize=[16,9])
plt.tight_layout()
ax = plt.subplot(1,1,1)



for cv in CVs:
    dX,Keq,Kf = find_sigma(cv)
    df = pd.read_csv(cv,header=None)
    #df[1] = df[1] / np.sqrt(sigma)
    #Convert the flux to dimensional current
    #df[1] = df[1] * 3.1415926*0.005641895835478*96485*1e-9*1.0*1e6
    #df[0] = df[0] / 96485 * 8.314*298.0 *1e3
    
    df.plot(x=0,y=1,ax=ax,linewidth=linewidth,color=tuple(colors[index]),label=f'$\\nu={dX:.2E}V/s$=,$k_{{eq}}={Keq:.2E}M$,$k_{{f}}={Kf:.2E} s^{{-1}}$')
    index += 1


for cv in ExperimentalCVs:
    df = pd.read_csv(cv,header=None)
    df.plot(x=0,y=1,ax=ax,linewidth=linewidth,label=cv)




ax.legend(loc=0,fontsize='medium') 

ax.tick_params(labelsize='large')
ax.set_xlabel(r"Potential,$\theta$", fontweight = "bold",fontsize='large')
ax.set_ylabel(r"Flux,$J$", fontweight = "bold",fontsize='large')


ax.annotate("", xy=(10, ax.get_ylim()[1]*0.1), xytext=(20, ax.get_ylim()[1]*0.1),arrowprops=dict(facecolor='black', shrink=0.05))

fig.savefig('All.png',bbox_inches='tight',dpi=400)



for CV in CVs:
    sigma,Kf,Kb = find_sigma(CV)
    df = pd.read_csv(CV, header=None)
    #df[1] = df[1] / np.sqrt(sigma)

    plt.figure(figsize=[16,12])
    plt.tight_layout()
    ax = plt.subplot(1,1,1)
    df_forward = df[:int(len(df)/2)]
    forward_peak = df_forward[0].iloc[df_forward[1].idxmin()]  #Forward peak potential
    df_backward = df[int(len(df)/2):]
    df_backward = df_backward.reset_index(drop=True)   #Use drop to discard the old index 
    backward_peak = df_backward[0].iloc[df_backward[1].idxmax()]  #Backward Peak Potential

    file_name = os.path.splitext(CV)[0]
    df.plot(x=0,y=1,ax=ax,linewidth=linewidth,label=f'$\\delta\\theta$={sigma},Kf={Kf:.2E},K0={Kb:.2E}')

    ax.set_xlabel("Theta", fontweight = "bold")
    ax.set_ylabel("Flux", fontweight = "bold")

    #Can plot peak current


    #Randles-Sevcik Prediction 
    forward_flux =  df_forward[1].min() 
    backward_flux = df_backward[1].max() 
    print(forward_flux/backward_flux)
    print(sigma,forward_flux)


    RevPrediction = -0.446 * np.sqrt(sigma)

    diff_from_pred = (forward_flux- RevPrediction)/RevPrediction *100 

    #ax.axhline(y=RevPrediction,linewidth=linewidth,linestyle='--',color='r',label='Randles-Sevcik Prediction')

    peak_sep = np.abs(forward_peak-backward_peak)

    alpha =  0.0#get_apparent_transfer_coefficient(CV)


    ax.set_title(f'Sigma={sigma:.2f},Kf={Kf:.2E},Kb={Kb:.2E} \n Diff from Reversible Randles-Sevcik Prediction: {diff_from_pred:.2f}% \n peak separation: {peak_sep:.2f} Forward peak and flux: {forward_peak:.2f}, {forward_flux:.5f}\n Backward peak and flux: {backward_peak:.2f} {backward_flux:.5f} alpha: {alpha:.2f}')
    ax.legend(loc=0) 

    




    plt.savefig(file_name+'.png',dpi=400)






for CV in ExperimentalCVs:
    df = pd.read_csv(CV, header=None)
    plt.figure(figsize=[16,12])
    plt.tight_layout()
    ax = plt.subplot(1,1,1)
    df_forward = df[:int(len(df)/2)]
    forward_peak = df_forward[0].iloc[df_forward[1].idxmax()]  #Forward peak potential
    df_backward = df[int(len(df)/2):]
    df_backward = df_backward.reset_index(drop=True)   #Use drop to discard the old index 
    backward_peak = df_backward[0].iloc[df_backward[1].idxmin()]  #Backward Peak Potential

    file_name = os.path.splitext(CV)[0]
    df.plot(x=0,y=1,ax=ax,linewidth=linewidth,label=CV)

    ax.set_xlabel("Theta", fontweight = "bold")
    ax.set_ylabel("Flux", fontweight = "bold")

    #Can plot peak current


    #Randles-Sevcik Prediction 
    forward_flux =  df_forward[1].max()
    backward_flux = df_backward[1].min()
    print(forward_flux/backward_flux)
    print(sigma,forward_flux)

    RevPrediction = 0.446 * np.sqrt(1198.257)

    diff_from_pred = (forward_flux- RevPrediction)/RevPrediction *100 

    ax.axhline(y=RevPrediction,linewidth=linewidth,linestyle='--',color='r',label='Randles-Sevcik Prediction')


    peak_sep = np.abs(forward_peak-backward_peak)

    alpha =  get_apparent_transfer_coefficient(CV)


    ax.set_title(file_name+"\n"+ f'forward peak flux: {forward_flux:.2f}, backward peak flux: {backward_flux:.2f}\n Diff from Reversible Randles-Sevcik Prediction: {diff_from_pred:.2f}% \n peak separation: {peak_sep:.2f} Forward peak: {forward_peak:.2f} Backward peak: {backward_peak:.2f} alpha: {alpha:.2f}')
    ax.legend(loc=0) 

    




    plt.savefig(file_name+'.png',dpi=400)


