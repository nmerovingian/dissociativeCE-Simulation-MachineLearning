import getpass
if getpass.getuser()=='RGCGroup':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #Run with CPU only
import pandas as pd
import numpy as np
from keras.layers import Dense,BatchNormalization,Dropout
from keras.models import Sequential
from keras.utils import plot_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import time
import os 

from helper import find_csv,find_sigma

import csv
linewidth = 3
fontsize = 14
figsize = [10,8]
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


import random
random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
np.random.seed(1)



def read_feature(file_name = 'Analysis.csv'):
    data = pd.read_csv(file_name)
    data = data.fillna(0.0)
    


    features = data[['Log10scanRate','Log10keq','Log10kf']]
    targets = data[['peak flux','half peak potential']]



    return features,targets

def create_model(data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
    model = Sequential()
    model.add(Dense(256,input_dim =data.shape[1] ,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(output_shape,activation='linear'))
    model.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error','mean_absolute_error'])
    return model




"""def create_model(data=None,optimizer='Adam',loss='mean_absolute_error'):
    model = Sequential()
    model.add(Dense(32,input_dim =data.shape[1] ,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16,activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error','mean_absolute_error'])
    return model"""

def extract(df,sigma):

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
    return[forward_flux,half_peak_potential]


    """
    #fig = plt.figure()
    df.plot(x='Theta',y='Flux',label='Cyclic Voltammogram')
    plt.scatter(range1,points1,label='Forward Scan Features')
    plt.scatter(range3,points3,label='Reverse Scan Features')
    plt.legend(loc=0)
    plt.show()
    time.sleep(1)
    #plt.close(fig)"""


if __name__ == "__main__":
    start = time.perf_counter()

    features,targets = read_feature()
    data_train,data_test,target_train,target_test = train_test_split(features,targets,test_size=0.2)

    """
    print(data.dtypes)
    print(target.head())
    print(data.head())
    """


    
    #Fit with NN
    model = create_model(data=data_train,output_shape=targets.shape[1])
    plot_model(model,to_file='Predict Flux and Half Peak Potential.png',show_shapes=True,show_layer_names=True,dpi=400)
    epochs=20000
    if not os.path.exists('./weights/Predict Flux and half peak potential.h5'):
        model.fit(data_train,target_train,epochs=epochs,batch_size=32,validation_split=0.2,verbose=1)
        model.save_weights('./weights/Predict Flux and half peak potential.h5')
    else:
        model.load_weights('./weights/Predict Flux and half peak potential.h5')
    pred = model.predict(data_test)
    print(pred[0])
    print('predicted value',pred,'actual value',target_test)


    #Fit with RandomForest
    RFregressor = RandomForestRegressor()
    RFregressor.fit(data_train,target_train)
    RFPredicted = RFregressor.predict(data_test)

    #Fit with LinearRegression
    regressor = LinearRegression()
    regressor.fit(data_train,target_train)
    LRPredicted = regressor.predict(data_test)
    """
    #Fit with XGBRegressor
    xgbregreesor = XGBRegressor()
    
    xgbregreesor.fit(data_train,target_train.iloc[:,0])
    XGBpredicted = xgbregreesor.predict(data_test)"""
    


    NNerror = mean_squared_error(target_test,pred)
    RFerror = mean_squared_error(target_test,RFPredicted)
    LRerror = mean_squared_error(target_test,LRPredicted)
    NNABSerror = mean_absolute_error(target_test,pred)
    RFABSerror = mean_absolute_error(target_test,RFPredicted)
    LRABSerror = mean_absolute_error(target_test,LRPredicted)
    #XGBerror = mean_squared_error(target_test,XGBpredicted)
    #XGBABSerror = mean_absolute_error(target_test,XGBpredicted)


    print('NN Error',NNerror,NNABSerror)
    print('Random Forest Error',RFerror,RFABSerror)
    print('Linear Regression error',LRerror,LRABSerror)
    #print('XGB Predicted error',XGBerror,XGBABSerror)






    fig = plt.figure(figsize=(16,9))
    ax1 = fig.add_subplot(2,2,1)
    plt.scatter(pred.reshape(-1),target_test)
    x = np.linspace(-15,10,100)
    plt.title(f'Neural Network, Train {epochs} times')
    plt.plot(x,x,'-r')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual Value')





    ax2 = fig.add_subplot(2,2,2)
    plt.scatter(RFPredicted.reshape(-1),target_test)
    x = np.linspace(-15,10,100)
    plt.title('Random Forest Prediction')
    plt.plot(x,x,'-r')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual Value')


    ax3 = fig.add_subplot(2,2,3)
    plt.scatter(LRPredicted.reshape(-1),target_test)
    x = np.linspace(-15,10,100)
    plt.title('Linear Prediction')
    plt.plot(x,x,'-r')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual Value')

    """
    ax4 = fig.add_subplot(2,2,4)
    plt.scatter(XGBpredicted.reshape(-1),target_test)
    x = np.linspace(-15,10,100)
    plt.title('XGBoost Prediction')
    plt.plot(x,x,'-r')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual Value')
    print('time consuming:',time.perf_counter()-start)
    """
    fig.savefig('Four  Model Results')

    plt.show()


    print()

    #visualization of one test sample 

    path_to_dir = './Test Data'
    Test_CVs=find_csv(path_to_dir=path_to_dir)

    with open('test_summary.csv',newline='',mode='w') as writer:
        csvwriter = csv.writer(writer)
        csvwriter.writerow(['log10scanRate','log10keq','log10kf','Flux','Predicted Flux','Half Peak Potential','Predicted Half Peak Potential','Predicted Flux Error','Predicted Half Peak Potential Error'])
        fig_flux_all,ax_flux_all = plt.subplots(figsize=(16,9))
        fig_potential_all,ax_potential_all = plt.subplots(figsize=(16,9))
        for index, test_cv in enumerate(Test_CVs):

            print(f'Testing {index} of {len(Test_CVs)-1} ')
            sigma,keq,kf = find_sigma(test_cv)
            filename  = test_cv.replace('.csv','')


            #if kf/keq > 1e9:
            #    continue




            df = pd.read_csv(f'{path_to_dir}/{test_cv}',header=None)

            #if df[1].min()>-0.01:
            #    continue
            #df.iloc[:,1] = df.iloc[:,1] + 0.001*np.random.randn(df.shape[0]) + df.iloc[:,1]*1e-3*np.random.randn(df.shape[0])

            range_all = extract(df.copy(),sigma)


            fig = plt.figure(figsize=(16,18))
            ax = plt.subplot(2,1,1)

            test_feature = np.array([[sigma,keq,kf]])
            test_feature = np.log10(test_feature)
            target = model.predict(test_feature)
            print(target.shape)

            error = np.average((target[0]-range_all)/range_all)
            csvwriter.writerow([np.log10(sigma),np.log10(keq),np.log10(kf),range_all[0],target[0,0],range_all[1],target[0,1],((target[0]-range_all))[0],((target[0]-range_all))[1]])
            #csvwriter.writerow([sigma,np.log10(keq),np.log10(kf),error,list(range_all)])
            #csvwriter.writerow([sigma,np.log10(keq),np.log10(kf),error,list(target[0])])

            plt.plot(np.arange(-10,10),np.arange(-10,10))
            plt.scatter(target[0,0],range_all[0],label='Peak Flux',color='r')
            plt.scatter(target[0,1],range_all[1],label='Half Peak Potential',color='b')
            plt.title('Test Results: Neural Network',fontsize='large')

            ax.legend(loc=0,fontsize='large') 
            ax.tick_params(labelsize='large')
            ax.set_xlabel(r"Predicted Values", fontweight = "bold",fontsize='large')
            ax.set_ylabel(r"True Values", fontweight = "bold",fontsize='large')
            #plt.show()

            ax = plt.subplot(2,1,2)
            RFtarget = RFregressor.predict(test_feature)
            plt.plot(np.arange(-10,10),np.arange(-10,10))
            plt.scatter(RFtarget[0,0],range_all[0],label='Peak Flux',color='r')
            plt.scatter(RFtarget[0,1],range_all[1],label='Half Peak Potential',color='b')
            plt.title('Test Results: Random Forest',fontsize='large')
            ax.legend(loc=0,fontsize='large') 
            ax.tick_params(labelsize='large')
            ax.set_xlabel(r"Predicted Values", fontweight = "bold",fontsize='large')
            ax.set_ylabel(r"True Values", fontweight = "bold",fontsize='large')
            fig.savefig(f'{path_to_dir}/{filename} Prediction.png')
            plt.close()
            #plt.show()



            ax_flux_all.scatter(target[0,0],range_all[0],label='Peak Flux',color='r')
            ax_potential_all.scatter(target[0,1],range_all[1],label='Half Peak Potential',color='b')
        
        ax_flux_all.plot(np.arange(-1.0,1.0),np.arange(-1.0,1.0))
        ax_flux_all.set_xlabel('Predicted Values')
        ax_flux_all.set_ylabel('True Values')
        ax_potential_all.plot(np.arange(-10.0,10.0),np.arange(-10.0,10.0))
        ax_potential_all.set_xlabel('Predicted Values')
        ax_potential_all.set_ylabel('True Values')
        fig_flux_all.savefig(f'{path_to_dir}/Flux All.png',dpi=400)
        fig_potential_all.savefig(f'{path_to_dir}/Potential All.png',dpi=400)
    writer.close()
        #Visualization of one test sample
        # = find_csv()








    