from helper import find_csv,find_sigma
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
np.random.seed(1)

#This is the path to training data directory. It's where the raw voltammogram files for training is saved.
# If you are usingh python for training, then the possible path could be:
#path_to_train_dir = "./Simulation Python/Data"
# If you are usingh C++ for training, then the possible path could be:
#path_to_train_dir = "./Simulation C++/Data"

path_to_train_dir = "./Training Data"

add_noise = False
noise_level = 0.0


def extract(file_name,dataframe,columns):
    df = pd.read_csv(f'{path_to_train_dir}/{file_name}',header=None)


    if add_noise:
        df.iloc[:,1] = df.iloc[:,1] + noise_level*np.random.randn(df.shape[0])
        df.iloc[:,1] = savgol_filter(x=df.iloc[:,1], window_length=51, polyorder=5)


    sigma,keq,kf = find_sigma(file_name)

    df.columns = ['Theta','Flux']


    cv_forward = df[:int(len(df)/2)]
    cv_backward = df[int(len(df)/2):]

    cv_backward = cv_backward.reset_index(drop=True)   #Use drop to discard the old index 

    forward_peak_flux = cv_forward['Flux'].min()
    forward_peak_potential = cv_forward['Theta'].iloc[cv_forward['Flux'].idxmin()]

    backward_peak_flux = cv_backward['Flux'].max()
    backward_peak_potential = cv_backward['Theta'].iloc[cv_backward['Flux'].idxmax()]

    phase1 = cv_forward[:cv_forward['Flux'].idxmin()]
    phase3 = cv_backward[:cv_backward['Flux'].idxmax()]

    points1 = np.linspace(0.01,1,num=100)*forward_peak_flux
    points3= cv_backward['Flux'].iloc[ 0]+np.linspace(0.01,1,num=100)*(np.abs(backward_peak_flux)+np.abs(cv_backward['Flux'].iloc[ 0]))


    range1 = np.array([])
    for point in points1:
        theta = phase1['Theta'].iloc[(phase1['Flux']-point).abs().argsort()[0]]
        range1 = np.append(range1,theta)

    range3 = np.array([])
    for point in points3:
        theta = phase3['Theta'].iloc[(phase3['Flux']-point ).abs().argsort()[0]]
        range3 = np.append(range3,theta)

    range_all = np.append(range1,points1)
    range_all = np.append(range_all,range3)
    range_all = np.append(range_all,points3)

    range_all = np.append(np.log10(np.array([sigma,keq,kf])),range_all)


    range_all = pd.Series(range_all,index=dataframe.columns)

    # A optional script to visualize the features extracted.
    """
    #fig = plt.figure()
    df.plot(x='Theta',y='Flux',label='Cyclic Voltammogram')
    plt.scatter(range1,points1,label='Forward Scan Features')
    plt.scatter(range3,points3,label='Reverse Scan Features')
    plt.legend(loc=0)
    plt.show()
    time.sleep(1)
    #plt.close(fig)"""
    return range_all












if __name__ == "__main__":
    file_names = find_csv(path_to_train_dir)
    columns = np.array(['Log10scanRate','Log10keq','Log10kf'])
    forward = np.array(['forward potential' + str(x) for x in np.linspace(0.01,1.0,100)])
    forward_flux = np.array(['forward flux' + str(x) for x in np.linspace(0.01,1.0,100)])
    backward = np.array(['backward potential' +str(x) for x in np.linspace(0.01,1.0,100)])
    backward_flux = np.array(['backward flux' +str(x) for x in np.linspace(0.01,1.0,100)])
    columns = np.append(columns,forward)
    columns = np.append(columns,forward_flux)
    columns = np.append(columns,backward)
    columns = np.append(columns,backward_flux)
    print(columns)
    DF  = pd.DataFrame([],columns=columns,dtype=np.float64)
    print(DF.head())
    for file_name in file_names:
        print(file_name)
        if file_name == 'features.csv':
            continue
        series = extract(file_name,DF,columns)
        DF =DF.append(series,ignore_index=True)


    DF.to_csv('features.csv',index=False)

