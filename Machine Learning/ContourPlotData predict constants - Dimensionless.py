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


df = pd.read_csv('Analysis.csv')
df = df.drop('Log10scanRate',axis=1)
df['Predicted log10kf'] = df[['Log10kf']]
df['Predicted log10keq'] = df[['Log10keq']]
df = df.set_index(['Log10kf','Log10keq'])

df = df.sort_values(by=['Log10kf','Log10keq'],ascending=True)


uniquekfvalues = len(df.index.get_level_values('Log10kf').unique())
uniquekeqvalues = len(df.index.get_level_values('Log10keq').unique())


df_test = pd.read_csv('Predict Constants.csv')
df_test = df_test.drop('log10scanRate',axis=1)

df_test = df_test.set_index(['log10kf','log10keq'])

df_test = df_test.sort_values(by=['log10kf','log10keq'],ascending=True)

uniquekfvalues_test = len(df_test.index.get_level_values('log10kf').unique())
uniquekeqvalues_test = len(df_test.index.get_level_values('log10keq').unique())




for i in df.index:
        if pow(10,i[0])/pow(10,i[1]) > 1.0e9:
                df.loc[i,'Predicted log10keq'] = np.nan
                df.loc[i,'Predicted log10kf'] = np.nan




for i in df_test.index:
        if pow(10,i[0])/pow(10,i[1]) > 1.0e9:
                df_test.loc[i,'Predicted log10keq'] = np.nan
                df_test.loc[i,'Predicted log10kx'] = np.nan
        else:

                if (abs(df_test.loc[i,'Predicted log10keq Error']) > error_tolerance).to_list()[0]:
                        df_test.loc[i,'log10keq'] = np.nan
                if (abs(df_test.loc[i,'Predicted log10kf Error']) > error_tolerance).to_list()[0]:
                        df_test.loc[i,'log10kf'] = np.nan


Predicted_log10keq = df['Predicted log10keq'].to_numpy().reshape([uniquekfvalues,uniquekeqvalues])

Predicted_log10kf = df['Predicted log10kf'].to_numpy().reshape([uniquekfvalues,uniquekeqvalues])

Predicted_log10keq_test = df_test['Predicted log10keq'].to_numpy().reshape([uniquekfvalues_test,uniquekeqvalues_test])

Predicted_log10kf_test = df_test['Predicted log10kf'].to_numpy().reshape([uniquekfvalues_test,uniquekeqvalues_test])

Kf = df.index.get_level_values('Log10kf').unique().to_numpy()
Keq = df.index.get_level_values('Log10keq').unique().to_numpy()


Kf_test = df_test.index.get_level_values('log10kf').unique().to_numpy()
Keq_test = df_test.index.get_level_values('log10keq').unique().to_numpy()


X, Y = np.meshgrid(Kf,Keq)

X_test,Y_test = np.meshgrid(Kf_test,Keq_test)

Predicted_log10keq_data = pd.DataFrame(Predicted_log10keq.T,columns = Kf,index=Keq)
Predicted_log10keq_data.to_csv('dimensional predicting keq.csv')

Predicted_log10kf_data = pd.DataFrame(Predicted_log10kf.T,columns = Kf,index=Keq)
Predicted_log10kf_data.to_csv('dimensional predicting kf.csv')
"""
##########################################################################
fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig)

ax.plot_wireframe(X,Y,Predicted_log10keq.T)
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
df_prediction =  df_prediction[['log10kf','log10keq','Predicted log10keq','Predicted log10keq Error']]
df_prediction_within_tolerance = (df_prediction['Predicted log10keq Error'].abs()<error_tolerance).sum()
print(df_prediction_within_tolerance,df_prediction_within_tolerance/len(df_prediction))
array_prediction = df_prediction[(df_prediction['Predicted log10keq Error']>0.0) & (df_prediction['Predicted log10keq Error']<error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='r',alpha=0.5)
array_prediction = df_prediction[(df_prediction['Predicted log10keq Error']<0.0) & (df_prediction['Predicted log10keq Error']>-error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='b',alpha=0.5)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.zaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
#ax.set_title(f'{df_prediction_within_tolerance/len(df_prediction):.2%} of testing data points are within {error_tolerance:.2%} error')

#Make legends with dummpy points
for a in [0.001,0.01,0.02,0.03,0.04,0.05,0.1]:
        ax.scatter([],[],c='k',alpha=0.5,s=a*3000,label=f'{a:.2%}')
ax.legend(scatterpoints=1,frameon=False,labelspacing=1,loc='center right')



plt.show()
#####################################################################
"""

######################################################################################################################################################################
fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig)
cmap = cm.viridis
lev = np.arange(-8.02,-2.98,0.02)
norml = colors.BoundaryNorm(lev, 512)
CS=ax.plot_surface(X,Y,Predicted_log10keq.T,cmap=cmap,norm=norml,alpha=0.9)
ax.view_init(45, 45)  #Elevation Azimuth
ax.set_zlim(top=7)
fig.colorbar(CS,label=r"$log{{10}}(\kappa_{{eq}})$",shrink=0.75)
ax.contour(X,Y,Predicted_log10keq.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.zaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log_{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
fig.savefig('Dimensional log10keq 3D.png',dpi=400)
######################################################################################################################################################################

######################################################################################################################################################################
fig = plt.figure(figsize=(20,9))
fig.tight_layout()
gs = gridspec.GridSpec(1,2,width_ratios=[4,1],wspace=0.2)
ax = fig.add_subplot(gs[0],projection='3d')
cmap = cm.viridis
lev = np.arange(-8.02,-2.98,0.02)
norml = colors.BoundaryNorm(lev, 512)
CS=ax.plot_surface(X,Y,Predicted_log10keq.T,cmap=cmap,norm=norml,alpha=0.9)
#ax.plot_surface(X_test,Y_test,Predicted_log10keq_test.T,color='k',alpha=0.2)
ax.view_init(45, 45)  #Elevation Azimuth
ax.set_zlim(top=5)
cbax = fig.add_axes([0.08,0.1,0.02,0.8])
fig.colorbar(CS,label=r'Predicted $log{{10}}(\kappa_{{eq}})$',shrink=0.75,cax=cbax,format = format_func_dimensionless_keq)
ax.contour(X,Y,Predicted_log10keq.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
df_prediction =  df_prediction[['log10kf','log10keq','Predicted log10keq','Predicted log10keq Error']]
df_prediction_within_tolerance = (df_prediction['Predicted log10keq Error'].abs()<error_tolerance).sum()
print(df_prediction_within_tolerance,df_prediction_within_tolerance/len(df_prediction))
array_prediction = df_prediction[(df_prediction['Predicted log10keq Error']>0.0) & (df_prediction['Predicted log10keq Error']<error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='r',alpha=0.5)
array_prediction = df_prediction[(df_prediction['Predicted log10keq Error']<0.0) & (df_prediction['Predicted log10keq Error']>-error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='b',alpha=0.5)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.zaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
#ax.set_title(f'{df_prediction_within_tolerance/len(df_prediction):.2%} of testing data points are within {error_tolerance:.2%} error')

#Make legends with dummpy points
for a in [0.001,0.01,0.02,0.03,0.04,0.05,0.1]:
        ax.scatter([],[],c='k',alpha=0.5,s=a*3000,label=f'{a:.2%}')
ax.legend(scatterpoints=1,frameon=False,labelspacing=1,loc='center right') 


# Add distribution plot 
ax = fig.add_subplot(gs[1])
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction,y='Predicted log10keq Error',bins=30,kde=True,ax=ax,binrange=(-0.5,0.5),stat='probability')
ax.set_ylim(-0.5,0.5)
ax.set_ylabel(r'Predicted $\kappa_{{eq}}$ Error, by Percentage')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major',length=7,width=2)
ax.tick_params(which='minor',length=5,color='r')
ax.yaxis.set_major_formatter(format_func_pct)


# add text to figure 
fig.text(0.15,0.75,'(a)',fontsize=40,fontweight='bold')
fig.text(0.65,0.75,'(b)',fontsize=40,fontweight='bold')


fig.savefig(f'Dimensional log10keq 3D with Predictions {error_tolerance} {df_prediction_within_tolerance/len(df_prediction):.4f}.png',dpi=400,bbox_inches = 'tight')
######################################################################################################################################################################

"""
fig,ax = plt.subplots(figsize=(16,9))
CS = ax.contourf(X,Y,Predicted_log10kf.T,cmap=cm.viridis)
fig.colorbar(CS,label='Predicted log10kf')
ax.set_xlabel(r'$Log_{{10}}(k_f,s^{{-1}})$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'$Log{{10}}(k_{{eq}},M)$',fontsize='large',fontweight='bold')
fig.savefig('Dimensional Predicted log10kf.png',dpi=400,bbox_inches='tight')
"""


######################################################################################################################################################################
fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig)
cmap = cm.viridis
lev = np.arange(-2.0,6.02,0.02)
norml = colors.BoundaryNorm(lev, 512)
CS=ax.plot_surface(X,Y,Predicted_log10kf.T,cmap=cmap,norm=norml)
ax.view_init(75, 90)  #Elevation Azimuth
ax.contour(X,Y,Predicted_log10kf.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
fig.colorbar(CS,label="Predicted log10kf",shrink=0.75)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.zaxis.set_major_formatter(format_func_dimensionless_kf)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
fig.savefig('Dimensional Predicted log10kf 3D.png',dpi=400)
############################################################################################################################################################################


#############################################################################################################################################################################
fig = plt.figure(figsize=(20,9))
fig.tight_layout()
gs = gridspec.GridSpec(1,2,width_ratios=[4,1],wspace=0.15)
ax = fig.add_subplot(gs[0],projection='3d')
cmap = cm.viridis
lev = np.arange(-2.02,6.02,0.02)
norml = colors.BoundaryNorm(lev, 512)
CS=ax.plot_surface(X,Y,Predicted_log10kf.T,cmap=cmap,norm=norml,alpha=0.9)
#ax.plot_surface(X_test,Y_test,Predicted_log10kf_test.T,color='k',alpha=0.2)
ax.view_init(75, 90)  #Elevation Azimuth
ax.set_zlim(top=6.02)
cbax = fig.add_axes([0.08,0.1,0.02,0.8])
fig.colorbar(CS,label=r'Predicted $log{{10}}(K_f)$',shrink=0.75,cax=cbax,format = format_func_dimensionless_kf)
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
ax.contour(X,Y,Predicted_log10kf.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
df_prediction = df_prediction[['log10kf','log10keq','Predicted log10kf','Predicted log10kf Error']]
df_prediction_within_tolerance = (df_prediction['Predicted log10kf Error'].abs()<error_tolerance).sum()
print(df_prediction_within_tolerance,df_prediction_within_tolerance/len(df_prediction))
array_prediction = df_prediction[(df_prediction['Predicted log10kf Error']>0.0) & (df_prediction['Predicted log10kf Error']<error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='r',alpha=0.5)
array_prediction = df_prediction[(df_prediction['Predicted log10kf Error']<0.0) & (df_prediction['Predicted log10kf Error']>-error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='b',alpha=0.5)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.zaxis.set_major_formatter(format_func_dimensionless_kf)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=20)
ax.set_ylabel(r'$log_{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=20)
#ax.set_title(f'{df_prediction_within_tolerance/len(df_prediction):.2%} of testing data points are within {error_tolerance:.2%} error')

#Make legends with dummpy points
for a in [0.001,0.01,0.02,0.03,0.04,0.05,0.1]:
        ax.scatter([],[],c='k',alpha=0.5,s=a*3000,label=f'{a:.2%}')
ax.legend(scatterpoints=1,frameon=False,labelspacing=1,loc='center right') 

# Add distribution plot 
ax = fig.add_subplot(gs[1])
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction,y='Predicted log10kf Error',bins=30,kde=True,ax=ax,binrange=(-0.5,0.5),stat='probability')
ax.set_ylim(-0.5,0.5)
ax.set_ylabel(r'Predicted $K_{{f}}$ Error, by Percentage')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major',length=7,width=2)
ax.tick_params(which='minor',length=5,color='r')
ax.yaxis.set_major_formatter(format_func_pct)


# add text to figure 
fig.text(0.15,0.75,'(a)',fontsize=40,fontweight='bold')
fig.text(0.65,0.75,'(b)',fontsize=40,fontweight='bold')

fig.savefig(f'Dimensional Predicted log10kf 3D with Predictions {error_tolerance} {df_prediction_within_tolerance/len(df_prediction):.4f}.png',dpi=400,bbox_inches = 'tight')
###########################################################################################################################





############################################################################################################################
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(2,1,1)
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['Predicted log10kf Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='probability')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $log_{{10}}(K_f)$ Prediction')
ax.xaxis.set_major_formatter(ticker.PercentFormatter())


ax = fig.add_subplot(2,1,2)
df_prediction = pd.read_csv('Predict Constants.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction['Predicted log10keq Error'],bins=100,kde=True,ax=ax,binrange=(-0.5,0.5),stat='probability')
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Percentage Error of $log{{10}}(\kappa_{{eq}})$ Prediction')
ax.xaxis.set_major_formatter(ticker.PercentFormatter())




