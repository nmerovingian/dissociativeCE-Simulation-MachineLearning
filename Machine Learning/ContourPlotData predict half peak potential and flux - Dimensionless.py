import pandas as pd
import numpy as np 
from matplotlib import gridspec, projections, pyplot as plt
from matplotlib import cm,colors
from mpl_toolkits.mplot3d import Axes3D
from helper import format_func_dimensionless_keq,format_func_dimensionless_kf
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn as sns


linewidth = 3
fontsize = 14
figsize = [10,8]


potential_error_tolerance = 0.01
flux_error_tolerance = 0.01


font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


df = pd.read_csv('Analysis.csv')
df = df.drop('Log10scanRate',axis=1)

df = df.set_index(['Log10kf','Log10keq'])

df = df.sort_values(by=['Log10kf','Log10keq'],ascending=True)


uniquekfvalues = len(df.index.get_level_values('Log10kf').unique())
uniquekeqvalues = len(df.index.get_level_values('Log10keq').unique())


df_test = pd.read_csv('test_summary.csv')
df_test = df_test.drop('log10scanRate',axis=1)

df_test = df_test.set_index(['log10kf','log10keq'])

df_test = df_test.sort_values(by=['log10kf','log10keq'],ascending=True)

uniquekfvalues_test = len(df_test.index.get_level_values('log10kf').unique())
uniquekeqvalues_test = len(df_test.index.get_level_values('log10keq').unique())




for i in df.index:
        if pow(10,i[0])/pow(10,i[1]) > 1.0e9:
                df.loc[i,'peak flux'] = np.nan
                df.loc[i,'half peak potential'] = np.nan




for i in df_test.index:
        if pow(10,i[0])/pow(10,i[1]) > 1.0e9:
                df_test.loc[i,'Flux'] = np.nan
                df_test.loc[i,'Half Peak Potential'] = np.nan
        else:
                #print(df_test.loc[i,'Flux']>-0.1)
                if (abs(df_test.loc[i,'Predicted Flux Error']) > flux_error_tolerance).to_list()[0]:
                        df_test.loc[i,'Flux'] = np.nan
                if (abs(df_test.loc[i,'Predicted Half Peak Potential Error']) > potential_error_tolerance).to_list()[0]:
                        df_test.loc[i,'Half Peak Potential'] = np.nan


peak_flux = df['peak flux'].to_numpy().reshape([uniquekfvalues,uniquekeqvalues])


half_peak_potential = df['half peak potential'].to_numpy().reshape([uniquekfvalues,uniquekeqvalues])

peak_flux_test = df_test['Flux'].to_numpy().reshape([uniquekfvalues_test,uniquekeqvalues_test])

half_peak_potential_test = df_test['Half Peak Potential'].to_numpy().reshape([uniquekfvalues_test,uniquekeqvalues_test])

Kf = df.index.get_level_values('Log10kf').unique().to_numpy()

Keq = df.index.get_level_values('Log10keq').unique().to_numpy()


Kf_test = df_test.index.get_level_values('log10kf').unique().to_numpy()
Keq_test = df_test.index.get_level_values('log10keq').unique().to_numpy()


X, Y = np.meshgrid(Kf,Keq)

X_test,Y_test = np.meshgrid(Kf_test,Keq_test)

peak_flux_data = pd.DataFrame(peak_flux.T,columns = Kf,index=Keq)
peak_flux_data.to_csv('dimensional peak_flux_data.csv')

half_peak_potential_data = pd.DataFrame(half_peak_potential.T,columns = Kf,index=Keq)
half_peak_potential_data.to_csv('dimensional_half_peak_potential_data.csv')

fig,ax = plt.subplots(figsize=(16,9))
CS = ax.contourf(X,Y,peak_flux.T,cmap=cm.viridis)
fig.colorbar(CS,label="Steady State Flux")
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
fig.savefig('Dimensionless Peak_Flux.png',dpi=400,bbox_inches='tight')

fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig)
cmap = cm.viridis
lev = np.arange(-1.1,0.0,0.02)
norml = colors.BoundaryNorm(lev, 256)
CS=ax.plot_surface(X,Y,peak_flux.T,cmap=cmap,norm=norml,alpha=0.9)
ax.view_init(30, 90)  #Elevation Azimuth
ax.set_zlim(top=0.01)

# Positioning of colorbar
colorbar_ax = fig.add_axes([0.1,0.1,0.03,0.8])  
cb = fig.colorbar(CS,label="Steady State Flux",shrink=0.75,cax=colorbar_ax)
ax.contour(X,Y,peak_flux.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_zlabel(r'Peak Flux, $J$',fontsize='large',fontweight='bold')
fig.savefig('Dimensional Peak_Flux 3D.png',dpi=400,bbox_inches = 'tight')


####################################################################################
fig = plt.figure(figsize=(20,9))
fig.tight_layout()
gs = gridspec.GridSpec(1,2,width_ratios=[4,1],wspace=0.2)
ax = fig.add_subplot(gs[0],projection='3d')
cmap = cm.viridis
lev = np.arange(-1.12,0.02,0.02)
norml = colors.BoundaryNorm(lev, 256)
CS=ax.plot_surface(X,Y,peak_flux.T,cmap=cmap,norm=norml,alpha=0.9)
#ax.plot_surface(X_test,Y_test,peak_flux_test.T,color='k',alpha=0.2)
ax.view_init(30, 90)  #Elevation Azimuth
ax.set_zlim(top=0.01)
# Add color bar 
cbax = fig.add_axes([0.08,0.1,0.02,0.8])
fig.colorbar(CS,label=r"Steady State Flux, $J$",shrink=0.75,cax=cbax)

ax.contour(X,Y,peak_flux.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
df_prediction = pd.read_csv('test_summary.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
df_prediction =  df_prediction[['log10kf','log10keq','Predicted Flux','Predicted Flux Error']]
df_prediction_within_tolerance = (df_prediction['Predicted Flux Error'].abs()<flux_error_tolerance).sum()
print(df_prediction_within_tolerance,df_prediction_within_tolerance/len(df_prediction))
array_prediction = df_prediction[(df_prediction['Predicted Flux Error']>0.0) & (df_prediction['Predicted Flux Error']<flux_error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='r',alpha=0.5)
array_prediction = df_prediction[(df_prediction['Predicted Flux Error']<0.0) & (df_prediction['Predicted Flux Error']>-flux_error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='b',alpha=0.5)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
#ax.set_title(f'{df_prediction_within_tolerance/len(df_prediction):.2%} of testing data points are within {flux_error_tolerance:.2%} error')

#Make legends with dummpy points
for a in [1e-3,2e-3,3e-3,4e-3,5e-3,1e-2]:
        ax.scatter([],[],c='k',alpha=1.0,s=a*3000,label=r'$\mid\epsilon_{{J}}\mid=$'+f'{a:.3f}')
ax.legend(scatterpoints=1,frameon=False,labelspacing=1,loc='center right') 



# Add distribution plot 
ax = fig.add_subplot(gs[1])
df_prediction = pd.read_csv('test_summary.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction,y='Predicted Flux Error',bins=30,kde=True,ax=ax,binrange=(-0.02,0.02),stat='probability')
ax.set_ylim(-0.020,0.02)
ax.set_ylabel(r'Predicted Flux Error, $\epsilon_{{J}}$')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major',length=7,width=2)
ax.tick_params(which='minor',length=5,color='r')

# add text to figure 
fig.text(0.15,0.75,'(a)',fontsize=40,fontweight='bold')
fig.text(0.65,0.75,'(b)',fontsize=40,fontweight='bold')
fig.savefig(f'Dimensional Peak_Flux 3D with Predictions {flux_error_tolerance}.png',dpi=400)
##########################################################################################



#########################################################################################
fig,ax = plt.subplots(figsize=(16,9))
CS = ax.contourf(X,Y,half_peak_potential.T,cmap=cm.viridis)
fig.colorbar(CS,label='Half Peak Potential')
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
fig.savefig('Dimensional Half Peak Potential.png',dpi=400,bbox_inches='tight')


fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig)
cmap = cm.viridis
lev = np.arange(-2.0,-0.02,0.02)
norml = colors.BoundaryNorm(lev, 256)
CS=ax.plot_surface(X,Y,half_peak_potential.T,cmap=cmap,norm=norml)
ax.view_init(75, 45)  #Elevation Azimuth
ax.contour(X,Y,half_peak_potential.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
fig.colorbar(CS,label="Half Peak Potential",shrink=0.75)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
fig.savefig('Dimensional Half Peak Potential 3D.png',dpi=400)

##################################################################################################################
fig = plt.figure(figsize=(20,9))
fig.tight_layout()
gs = gridspec.GridSpec(1,2,width_ratios=[4,1],wspace=0.2)
ax = fig.add_subplot(gs[0],projection='3d')
cmap = cm.viridis
lev = np.arange(-2.02,0.02,0.02)
norml = colors.BoundaryNorm(lev, 256)
CS=ax.plot_surface(X,Y,half_peak_potential.T,cmap=cmap,norm=norml,alpha=0.9)
#ax.plot_surface(X_test,Y_test,half_peak_potential_test.T,color='k',alpha=0.2)
ax.view_init(75, 45)  #Elevation Azimuth
ax.set_zlim(top=0.01)
cbax = fig.add_axes([0.08,0.1,0.02,0.8])
fig.colorbar(CS,label=r"Half Peak Potential, $\theta$",shrink=0.75,cax=cbax)
df_prediction = pd.read_csv('test_summary.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
ax.contour(X,Y,half_peak_potential.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
df_prediction = df_prediction[['log10kf','log10keq','Predicted Half Peak Potential','Predicted Half Peak Potential Error']]
df_prediction_within_tolerance = (df_prediction['Predicted Half Peak Potential Error'].abs()<potential_error_tolerance).sum()
print(df_prediction_within_tolerance,df_prediction_within_tolerance/len(df_prediction))
array_prediction = df_prediction[(df_prediction['Predicted Half Peak Potential Error']>0.0) & (df_prediction['Predicted Half Peak Potential Error']<potential_error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='r',alpha=0.5)
array_prediction = df_prediction[(df_prediction['Predicted Half Peak Potential Error']<0.0) & (df_prediction['Predicted Half Peak Potential Error']>-potential_error_tolerance)].to_numpy()
ax.scatter(array_prediction[:,0],array_prediction[:,1],array_prediction[:,2],s=abs(array_prediction[:,3])*3000,color='b',alpha=0.5)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=10)
ax.set_zticks([-2.0,-1.5,-1.0,-0.5,0.0])
#ax.set_title(f'{df_prediction_within_tolerance/len(df_prediction):.2%} of testing data points are within {potential_error_tolerance:.2%} error')


#Make legends with dummpy points
for a in [1e-3,2e-3,3e-3,4e-3,5e-3,1e-2]:
        ax.scatter([],[],c='k',alpha=1.0,s=a*3000,label=r'$\mid\epsilon_{{\theta}}\mid$='+f'{a:.3f}')
ax.legend(scatterpoints=1,frameon=False,labelspacing=1,loc='top right') 


# Add distribution plot 
ax = fig.add_subplot(gs[1])
df_prediction = pd.read_csv('test_summary.csv')
df_prediction = df_prediction [(df_prediction['log10kf']-df_prediction['log10keq'])<=9]
sns.histplot(data = df_prediction,y='Predicted Half Peak Potential Error',bins=30,kde=True,ax=ax,binrange=(-0.05,0.05),stat='probability')
ax.set_ylim(-0.05,0.05)
ax.set_ylabel(r'Predicted Half Peak Potential Error, $\epsilon_{{\theta}}$')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major',length=7,width=2)
ax.tick_params(which='minor',length=5,color='r')

# add text to figure 
fig.text(0.15,0.75,'(a)',fontsize=40,fontweight='bold')
fig.text(0.7,0.85,'(b)',fontsize=40,fontweight='bold')


fig.savefig(f'Dimensional Half Peak Potential 3D with Predictions {potential_error_tolerance}.png',dpi=400)
##############################################################################################################################################





################################################################################################################################################
fig = plt.figure(figsize=(32,9),constrained_layout=True)
fig.tight_layout()
ax = fig.add_subplot(1,2,1,projection = '3d')
cmap = cm.viridis
lev = np.arange(-1.1,0.00,0.02)
norml = colors.BoundaryNorm(lev, 256)
CS=ax.plot_surface(X,Y,peak_flux.T,cmap=cmap,norm=norml,alpha=0.9)
ax.view_init(30, 90)  #Elevation Azimuth
ax.set_zlim(top=0.01)
# Positioning of colorbar
colorbar_ax = fig.add_axes([0.05,0.1,0.01,0.8])  
cb = fig.colorbar(CS,shrink=0.75,cax=colorbar_ax)
cb.set_label(label="Steady State Flux, $J$",weight='bold',size='large')
ax.contour(X,Y,peak_flux.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad = 30)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad = 30)

ax.tick_params(axis="both", labelsize=20)
ax.tick_params(axis='both', which='major', pad=20)
fig.text(0.1,0.9,'(a)',fontsize=40,fontweight='bold')


ax = fig.add_subplot(1,2,2,projection = '3d')
cmap = cm.viridis
lev = np.arange(-2.02,0.02,0.02)
norml = colors.BoundaryNorm(lev, 256)
CS=ax.plot_surface(X,Y,half_peak_potential.T,cmap=cmap,norm=norml)
ax.view_init(75, 45)  #Elevation Azimuth 75 ,45
ax.contour(X,Y,half_peak_potential.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)

# Positioning of colorbar
colorbar_ax = fig.add_axes([0.47,0.1,0.01,0.8])  
cb = fig.colorbar(CS,shrink=0.75,cax=colorbar_ax)
cb.set_label(label=r"Half Peak Potential, $\theta$",weight='bold',size='large')
ax.xaxis.set_major_formatter(format_func_dimensionless_kf)
ax.yaxis.set_major_formatter(format_func_dimensionless_keq)
ax.set_xlabel(r'$log_{{10}}(K_f)$',fontsize='large',fontweight='bold',labelpad = 30)
ax.set_ylabel(r'$log{{10}}(\kappa_{{eq}})$',fontsize='large',fontweight='bold',labelpad=30)
ax.set_zticks([-2.0,-1.5,-1.0,-0.5,0.0])


fig.text(0.52,0.9,'(b)',fontsize=40,fontweight='bold')


ax.tick_params(axis="both", labelsize=20)
ax.tick_params(axis='both', which='major', pad=15)
fig.savefig('Dimensional Peak Flux and Half Peak Potential 3D.png',dpi=400,bbox_inches='tight')
#######################################################################################################################
######################################################################













"""

Kf = df.index.get_level_values('Log10kf').unique().to_numpy()-3.0
Keq = df.index.get_level_values('Log10keq').unique().to_numpy()+3.0

X, Y = np.meshgrid(Kf,Keq)

peak_flux_data = pd.DataFrame(peak_flux.T,columns = Kf,index=Keq)
peak_flux_data.to_csv('peak_flux_data.csv')

half_peak_potential_data = pd.DataFrame(half_peak_potential.T,columns = Kf,index=Keq)
half_peak_potential_data.to_csv('half_peak_potential_data.csv')

fig,ax = plt.subplots(figsize=(16,9))
CS = ax.contourf(X,Y,peak_flux.T,cmap=cm.viridis)
fig.colorbar(CS,label="Steady State Flux")
ax.set_xlabel(r'$Log_{{10}}(K_f)$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'$Log{{10}}(K_{{eq}})$',fontsize='large',fontweight='bold')
fig.savefig('Peak_Flux.png',dpi=400,bbox_inches='tight')

fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig)
CS=ax.plot_surface(X,Y,peak_flux.T,cmap=cm.viridis,alpha=0.9)
ax.view_init(30, 45)  #Elevation Azimuth
fig.colorbar(CS,label="Steady State Flux")
ax.contour(X,Y,peak_flux.T,zdir='z',offset=ax.get_zlim()[0],cmap=cm.viridis)
ax.set_xlabel(r'$Log_{{10}}(K_f)$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'$Log{{10}}(K_{{eq}})$',fontsize='large',fontweight='bold')
plt.show()
fig.savefig('Peak_Flux 3D.png',dpi=400)


fig,ax = plt.subplots(figsize=(16,9))
CS = ax.contourf(X,Y,half_peak_potential.T,cmap=cm.viridis)
fig.colorbar(CS,label='Half Peak Potential')
ax.set_xlabel(r'$Log_{{10}}(K_f)$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'$Log{{10}}(K_{{eq}})$',fontsize='large',fontweight='bold')
fig.savefig('Half Peak Potential.png',dpi=400,bbox_inches='tight')


fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig)
CS=ax.plot_surface(X,Y,half_peak_potential.T,cmap=cm.viridis)
fig.colorbar(CS,label="Half Peak Potential")
ax.set_xlabel(r'$Log_{{10}}(K_f)$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'$Log{{10}}(K_{{eq}})$',fontsize='large',fontweight='bold')
fig.savefig('Half Peak Potential 3D.png',dpi=400)"""
