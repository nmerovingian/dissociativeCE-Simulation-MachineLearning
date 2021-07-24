from helper import find_csv,find_sigma,find_experimental_csv
from OxTafelSlope import get_apparent_transfer_coefficient
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import re
import numpy as np
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
#from multiprocessing import Lock
from threading import Lock
CVs = find_csv()
#CVs = CVs[::-1]
ExperimentalCVs = find_experimental_csv()
import csv
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

#fig = plt.figure(figsize=[16,9])
#plt.tight_layout()
#ax = plt.subplot(1,1,1)


def analyzeData(CV,lock):
    sigma,keq,kf = find_sigma(CV)

    df = pd.read_csv(CV, header=None)




    df_forward = df[:int(len(df)/2)]
    forward_peak = df_forward[0].iloc[df_forward[1].idxmin()]  #Forward peak potential
    df_backward = df[int(len(df)/2):]
    df_backward = df_backward.reset_index(drop=True)   #Use drop to discard the old index 
    backward_peak = df_backward[0].iloc[df_backward[1].idxmax()]  #Backward Peak Potential
    
    #Randles-Sevcik Prediction 
    forward_flux =  df_forward[1].min() 
    backward_flux = df_backward[1].max() 

    #Find the half peak potential
    half_peak_potential_index = (df_forward[1]-forward_flux/2.0).abs().argsort()[0]
    half_peak_potential = df_forward[0].iloc[half_peak_potential_index]

    print('half peak potential index',half_peak_potential_index,half_peak_potential)

    #transfer coefficient at half peak potential
    #print(df_forward[1].iloc[half_peak_potential_index+1],df_forward[1].iloc[half_peak_potential_index])
    transfer_coefficient_half_peak = (np.log(-df_forward[1].iloc[half_peak_potential_index+10]) - np.log(-df_forward[1].iloc[half_peak_potential_index])) / ((df_forward[0].iloc[half_peak_potential_index+1] - df_forward[0].iloc[half_peak_potential_index])*10)
    print('transfer coefficient',(np.log(-df_forward[1].iloc[half_peak_potential_index+1]) - np.log(-df_forward[1].iloc[half_peak_potential_index])),transfer_coefficient_half_peak)
    

    file_name = os.path.splitext(CV)[0]

    


    #Can plot peak current



    print(forward_flux/backward_flux)
    print(sigma,forward_flux)


    RevPrediction = -0.446 * np.sqrt(sigma)

    diff_from_pred = (forward_flux- RevPrediction)/RevPrediction *100 

    #ax.axhline(y=RevPrediction,linewidth=linewidth,linestyle='--',color='r',label='Randles-Sevcik Prediction')

    peak_sep = np.abs(forward_peak-backward_peak)

    alpha = 0.0# get_apparent_transfer_coefficient(CV)


    """
    plt.figure(figsize=[16,12])
    plt.tight_layout()
    ax = plt.subplot(1,1,1)
    df.plot(x=0,y=1,ax=ax,linewidth=linewidth,label=f'Sigma={sigma:.2E} V/s,keq={keq:.2E}M,kf={kf:.2E}s^-1')

    ax.set_xlabel("Theta", fontweight = "bold")
    ax.set_ylabel("Flux", fontweight = "bold")
    ax.set_title(f'Sigma={sigma:.2f} V/s,Kf={kf:.2E},keq={keq:.2E} \n Diff from Reversible Randles-Sevcik Prediction: {diff_from_pred:.2f}% \n peak separation: {peak_sep:.2f} Forward peak and flux: {forward_peak:.2f}, {forward_flux:.5f}\n Backward peak and flux: {backward_peak:.2f} {backward_flux:.5f} alpha: {alpha:.2f}')
    ax.legend(loc=0) 
    plt.savefig(file_name+'.png',dpi=400)
    plt.close()"""

    lock.acquire()
    with open('analysis.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([np.log10(sigma),np.log10(kf),np.log10(keq),forward_flux,half_peak_potential,transfer_coefficient_half_peak])
    lock.release()


if __name__ == "__main__": 
    lock = Lock()
    with open('analysis.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Log10scanRate','Log10kf','Log10keq','peak flux','half peak potential','half peak transfer coefficient'])


    for cv in CVs:
        analyzeData(cv,lock)
    