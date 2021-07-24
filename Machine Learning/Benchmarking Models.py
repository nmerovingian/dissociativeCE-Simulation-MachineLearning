import pandas as pd
import numpy as np 
from matplotlib import gridspec, projections, pyplot as plt
from matplotlib import cm,colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from helper import format_func_dimensionless_keq,format_func_dimensionless_kf,format_func_pct
import seaborn as sns

linewidth = 3
fontsize = 14
figsize = [10,8]


error_tolerance = 0.1


font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(16,19),gridspec_kw={'hspace': 0.5},sharey=True)


ax = axes[0][0]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['Predicted log10kf Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(K_f)$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'(a)',fontsize=20,fontweight='bold')

ax = axes[0][1]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['Predicted log10keq Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(\kappa_{{eq}})$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'(b)',fontsize=20,fontweight='bold')




ax = axes[1][0]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['LR Predicted log10kf Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(K_f)$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'(c)',fontsize=20,fontweight='bold')

ax = axes[1][1]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
df_prediction = df_prediction.replace([np.inf, -np.inf], np.nan).dropna(how="all")
df_prediction = df_prediction[(df_prediction['LR Predicted log10keq Error']<0.5) &(df_prediction['LR Predicted log10keq Error']>-0.5)]
sns.histplot(data = df_prediction['LR Predicted log10keq Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(\kappa_{{eq}})$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'(d)',fontsize=20,fontweight='bold')


ax = axes[2][0]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['RF Predicted log10kf Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(K_f)$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'(e)',fontsize=20,fontweight='bold')

ax = axes[2][1]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['RF Predicted log10keq Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(\kappa_{{eq}})$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'(f)',fontsize=20,fontweight='bold')


ax = axes[3][0]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['XG Predicted log10kf Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(K_f)$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'g)',fontsize=20,fontweight='bold')

ax = axes[3][1]
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['XG Predicted log10keq Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='count')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $(\kappa_{{eq}})$ Prediction')
ax.xaxis.set_major_formatter(format_func_pct)
ax.text(-0.45,130,'(h)',fontsize=20,fontweight='bold')

fig.savefig('Benchmarking Models.png',dpi=400,bbox_inches='tight')