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

df = pd.read_csv('predict voltammogram potential.csv')

print(len(df[df['forward scan current error']>0.01]))

fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(16,9))

ax = axes[0][0]

sns.histplot(df,x='forward scan potential error',kde=True,binrange=(-0.2,0.2),bins=40,ax=ax,stat='probability')
ax.set_xlabel(r'Forward Scan Potential Error, $\epsilon_{{\theta}}$',fontsize='large',fontweight='bold')
ax.set_xticks([-0.2,-0.1,0.0,0.1,0.2])
ax.text(-0.2,0.45,'(a)',fontsize=20,fontweight='bold')

ax = axes[0][1]
sns.histplot(df,x='forward scan current error',kde=True,binrange=(-0.025,0.025),bins=40,ax=ax,stat='probability')
ax.set_xlabel(r'Forward Scan Flux Error, $\epsilon_{{J}}$',fontsize='large',fontweight='bold')
ax.text(-0.025,0.32,'(c)',fontsize=20,fontweight='bold')

ax = axes[1][0]
sns.histplot(df,x='reverse scan potential error',kde=True,binrange=(-0.2,0.2),bins=40,ax=ax,stat='probability')
ax.set_xlabel(r'Reverse Scan Potential Error, $\epsilon_{{\theta}}$',fontsize='large',fontweight='bold')
ax.set_xticks([-0.2,-0.1,0.0,0.1,0.2])
ax.text(-0.2,0.45,'(b)',fontsize=20,fontweight='bold')
ax = axes[1][1]
sns.histplot(df,x='reverse scan current error',kde=True,binrange=(-0.025,0.025),bins=40,ax=ax,stat='probability')
ax.set_xlabel(r'Reverse Scan Flux Error, $\epsilon_{{J}}$',fontsize='large',fontweight='bold')
ax.text(-0.025,0.32,'(d)',fontsize=20,fontweight='bold')
fig.savefig('Predict Voltammogram NN Error Distribution.png',dpi=400,bbox_inches = 'tight')



fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(16,9))

ax = axes[0][0]

sns.histplot(df,x='RFforward scan potential error',kde=True,binrange=(-0.5,0.5),bins=40,ax=ax,stat='probability')
ax = axes[0][1]
sns.histplot(df,x='RFforward scan current error',kde=True,binrange=(-0.1,0.1),bins=40,ax=ax,stat='probability')
ax = axes[1][0]
sns.histplot(df,x='RFreverse scan potential error',kde=True,binrange=(-0.5,0.5),bins=40,ax=ax,stat='probability')
ax = axes[1][1]
sns.histplot(df,x='RFreverse scan current error',kde=True,binrange=(-0.1,0.1),bins=40,ax=ax,stat='probability')


fig.savefig('Predict Voltammogram RF Error Distribution.png')