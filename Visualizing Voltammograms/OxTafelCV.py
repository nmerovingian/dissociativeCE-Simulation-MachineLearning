from helper import find_csv,find_sigma
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import re
import numpy as np

linewidth = 4
fontsize = 20
figsize = [10,8]

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs
from matplotlib import cycler 
#plt.rcParams['axes.prop_cycle'] = cycler(linestyle=['-','-.','--',':','-','-.','--',':'],color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])

from helper import format_func_dimensionla_potential



def getTafelRegion(CV,start=0.05, end = 0.15):
    df = pd.read_csv(CV,header=None)
    cv_forward = df[:int(len(df)/2)]
    min_current = cv_forward[1].min()
    min_current_index = cv_forward[1].idxmin()

    cv_before_peak = cv_forward.iloc[:min_current_index]

    cv_tafel = cv_before_peak[(cv_before_peak[1]<start*min_current)&(cv_before_peak[1]>end*min_current)]
    flux = pd.DataFrame(cv_tafel[1])
    cv_tafel[1] = np.log(-cv_tafel[1])

    x = pd.DataFrame(cv_tafel[0])
    y= pd.DataFrame(cv_tafel[1])

    #X Theta
    #Y Log(J)

    Gradient = pd.DataFrame(columns=['Theta','flux','LnFlux','Gradient'])
    Gradient['Theta'] = x[0]
    Gradient['LnFlux'] = y
    Gradient['flux'] = flux
    for index, value in enumerate(Gradient['Theta']):
        if index < len(Gradient['Theta'])-2:
            Gradient['Gradient'].iloc[index] = (Gradient['LnFlux'].iloc[index+1] - Gradient['LnFlux'].iloc[index])/(Gradient['Theta'].iloc[index+1]-Gradient['Theta'].iloc[index])
            #print(index,Gradient['Gradient'].iloc[index])
        else:
            Gradient['Gradient'].iloc[index] = Gradient['Gradient'].iloc[index-1]

    Gradient['Gradient'] = -Gradient['Gradient']

    Gradient_name = 'Gradient' + CV
    Gradient.to_csv(Gradient_name,index=False)



def plotTafelRegion(CV):
    if 'One Electron Reduction' in CV:
        Gradient = pd.read_csv(CV)
        offset = Gradient.iloc[0,0]
        Gradient['Theta'] = Gradient['Theta'] - offset

        Transfer_coefficient_at_5pct = Gradient['Gradient'][0]
        Transfer_coefficient_at_30pct = Gradient['Gradient'].iloc[-1]

        Gradient.plot(x='Theta',y='Gradient',ax=ax,linewidth = linewidth, label = f'One Electron Reduction',ls='--',color='k')
    else:
        Gradient = pd.read_csv(CV)
        offset = Gradient.iloc[0,0]
        Gradient['Theta'] = Gradient['Theta'] - offset
        sigma,Kf,Keq = find_sigma(CV)
        Transfer_coefficient_at_5pct = Gradient['Gradient'][0]
        Transfer_coefficient_at_30pct = Gradient['Gradient'].iloc[-1]
        global colorindex 
        Gradient.plot(x='Theta',y='Gradient',ax=ax,linewidth = linewidth, color = tuple(colors[colorindex]),label = f'$\\nu = {sigma/1239.60592743:.2E} mV/s$')
        colorindex +=1




if __name__ == "__main__":
    fig = plt.figure(figsize=[20,12])
    plt.tight_layout()
    ax = plt.subplot(1,1,1)
    start = 0.05
    end = 0.30

    CVs = find_csv()
    colors = cm.viridis(np.linspace(0,1,len(CVs)))
    CVs +=['One Electron Reduction.csv']
    CVs = CVs[::-1]

    colorindex = 0 
    for CV in CVs:
        if 'Gradient' in CV:
            continue
        getTafelRegion(CV,start,end)

    #Now plot Gradient
    CVs = find_csv()
    CVs += ['GradientOne Electron Reduction.csv']
    for CV in CVs:
        if 'Gradient' in CV:
            plotTafelRegion(CV)
    
ax.set_xlabel(r"Potential,$\theta$", fontweight = "bold")
ax.set_ylabel("Apparent Transfer Coefficient, $\\alpha$", fontweight = "bold")
#ax.set_ylim(0,ax.get_ylim()[1]*1.2)
#ax.set_title(f'Gradient from {start:.2%} to {end:.2%} of peak current')

# format x axis
#ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_dimensionla_potential))


ax.annotate("", xy=(-0.5, ax.get_ylim()[1]*0.95), xytext=(-0.0, ax.get_ylim()[1]*0.95),arrowprops=dict(facecolor='black', shrink=0.05))
fig.savefig('GradientAll.png',dpi=400,bbox_inches='tight')

fig.savefig(r'C:\Users\nmero\OneDrive - Nexus365\Paper3\Paper Figures\Fig8.png',bbox_inches='tight',dpi=400)

