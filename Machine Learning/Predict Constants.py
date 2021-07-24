import getpass
if getpass.getuser()=='RGCGroup':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #Run with CPU only
import pandas as pd
import numpy as np
from keras.layers import Dense,BatchNormalization,Dropout,LSTM,Input
from keras.layers.merge import concatenate
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential,Model
from keras.utils import plot_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.signal import savgol_filter 
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import time
import os

from helper import find_csv,find_sigma

import csv

from ann_visualizer.visualize import ann_viz

import random
random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
np.random.seed(1)

linewidth = 3
fontsize = 14
figsize = [10,8]
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs
path_to_test_dir = './Test Data'


add_noise = False
noise_level = 1e-4



def read_feature(file_name = 'features.csv'):
    data = pd.read_csv(file_name)
    data = data.fillna(0.0)


    targets = data[['Log10scanRate','Log10keq','Log10kf']]
    features = data.drop(['Log10scanRate','Log10keq','Log10kf'],axis=1)



    return features,targets
"""
def create_model(data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
    model = Sequential()
    model.add(Dense(256,input_dim =data.shape[1] ,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(output_shape,activation='linear'))
    model.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error','mean_absolute_error'])
    return model
"""

# Multiheaded DNN
def create_model(data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
    # head1 
    inputs1 = Input(shape=(data.shape[1],))
    dnn11 = Dense(256,activation='relu')(inputs1)
    dnn12 = Dense(128,activation='relu')(dnn11)
    dnn13 = Dense(64,activation='relu')(dnn12)
    dnn14 = Dense(32,activation='linear')(dnn13)
    #head2
    inputs2 = Input(shape=(data.shape[1],))
    dnn21 = Dense(256,activation='relu')(inputs2)
    dnn22 = Dense(128,activation='relu')(dnn21)
    dnn23 = Dense(64,activation='relu')(dnn22)
    dnn24 = Dense(32,activation='linear')(dnn23)

    # merge
    merged = concatenate([dnn14,dnn24])
    outputs = Dense(output_shape)(merged)

    model = Model(inputs=[inputs1,inputs2],outputs=outputs)
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

    df.columns = ['Theta','Flux']

    #df['Flux'] = df['Flux'] / np.sqrt(sigma)

    cv_forward = df[:int(len(df)/2)]
    cv_backward = df[int(len(df)/2):]
    #print(df.head())
    cv_backward = cv_backward.reset_index(drop=True)   #Use drop to discard the old index 
    #print(cv_backward.head())
    forward_peak_flux = cv_forward['Flux'].min()
    forward_peak_potential = cv_forward['Theta'].iloc[cv_forward['Flux'].idxmin()]
    #print(forward_peak_flux)
    #print(forward_peak_potential)
    backward_peak_flux = cv_backward['Flux'].max()
    backward_peak_potential = cv_backward['Theta'].iloc[cv_backward['Flux'].idxmax()]
    #print(backward_peak_flux)
    #print(backward_peak_potential)
    phase1 = cv_forward[:cv_forward['Flux'].idxmin()]
    phase3 = cv_backward[:cv_backward['Flux'].idxmax()]
    #print(phase1.tail())
    #print(phase3.tail())
    points1 = np.linspace(0.01,1,num=100)*forward_peak_flux
    points3= cv_backward['Flux'].iloc[ 0]+np.linspace(0.01,1,num=100)*(np.abs(backward_peak_flux)+np.abs(cv_backward['Flux'].iloc[ 0]))
    #print(points1)
    #print(points3)

    range1 = np.array([])
    for point in points1:
        theta = phase1['Theta'].iloc[(phase1['Flux']-point).abs().argsort()[0]]
        range1 = np.append(range1,theta)
    #print(range1)
    range3 = np.array([])
    for point in points3:
        theta = phase3['Theta'].iloc[(phase3['Flux']-point ).abs().argsort()[0]]
        range3 = np.append(range3,theta)
    #print(range3)
    range_all = np.append(range1,points1)
    range_all = np.append(range_all,range3)
    range_all = np.append(range_all,points3)
    #print(range_all)


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

def schedule(epoch,lr):
    if epoch <1000:
        return lr
    else:
        return lr*0.9999

if __name__ == "__main__":
    start = time.perf_counter()

    scheduler = LearningRateScheduler(schedule,verbose=0)
    features,targets = read_feature()
    data_train,data_test,target_train,target_test = train_test_split(features,targets,test_size=0.2)

    """
    print(data.dtypes)
    print(target.head())
    print(data.head())
    """
    """
    scaler = StandardScaler()
    scaler = scaler.fit(data_train)

    print(features.shape)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    """

    
    #Fit with NN
    
    model = create_model(data=data_train,output_shape=targets.shape[1])
    plot_model(model,to_file='Predict Constants.png',show_shapes=True,show_layer_names=True,dpi=400)
    epochs=10000
    if not os.path.exists('./weights/Predict Constants.h5'):
        model.fit([data_train,data_train],target_train,epochs=epochs,batch_size=16,validation_split=0.2,verbose=1,callbacks=[scheduler])
        model.save_weights('./weights/Predict Constants.h5')

    else:
        model.load_weights('./weights/Predict Constants.h5')


    print(model.summary())
    for layer in model.layers: print(layer.get_config(), layer.get_weights().shape)
    input()
    ann_viz(model,title='Method A',filename='test.gv')
    pred = model.predict([data_test,data_test])
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
    
    #Fit with XGBRegressor
    XGBregressors = []
    XGBpredicted = []
    for i in range(targets.shape[1]): 
        XGBregressors.append(XGBRegressor())
    
        XGBregressors[i].fit(data_train.to_numpy(),target_train.iloc[:,i].to_numpy())
        XGBpredicted.append( XGBregressors[i].predict(data_test.to_numpy()))

    XGBpredicted = np.stack(tuple(XGBpredicted),axis=1)
    


    NNerror = mean_squared_error(target_test,pred)
    RFerror = mean_squared_error(target_test,RFPredicted)
    LRerror = mean_squared_error(target_test,LRPredicted)
    NNABSerror = mean_absolute_error(target_test,pred)
    RFABSerror = mean_absolute_error(target_test,RFPredicted)
    LRABSerror = mean_absolute_error(target_test,LRPredicted)
    XGBerror = mean_squared_error(target_test,XGBpredicted)
    XGBABSerror = mean_absolute_error(target_test,XGBpredicted)

    print('Model,Mean suqred error, mean absolute error')
    print('NN Error',NNerror,NNABSerror)
    print('Random Forest Error',RFerror,RFABSerror)
    print('Linear Regression error',LRerror,LRABSerror)
    print('XGB Predicted error',XGBerror,XGBABSerror)






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

    
    ax4 = fig.add_subplot(2,2,4)
    plt.scatter(XGBpredicted.reshape(-1),target_test)
    x = np.linspace(-15,10,100)
    plt.title('XGBoost Prediction')
    plt.plot(x,x,'-r')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual Value')
    print('time consuming:',time.perf_counter()-start)
    
    fig.savefig('Four  Model Results')

    plt.show()


    print()

    #visualization of one test sample 


    Test_CVs=find_csv(path_to_dir=path_to_test_dir)

    with open('Predict Constants.csv',newline='',mode='w') as writer:
        csvwriter = csv.writer(writer)
        csvwriter.writerow(['log10scanRate','log10keq','log10kf','Predicted log10keq','Predicted log10keq Error','Predicted log10kf','Predicted log10kf Error','LR Predicted log10keq Error','LR Predicted log10kf Error','RF Predicted log10keq Error','RF Predicted log10kf Error','XG Predicted log10keq Error','XG Predicted log10kf Error'])
        #csvwriter.writerow(['scanRate,V/s','keq,M','log10kf,/s','Predicted log10keq','Predicted log10keq Error','Predicted log10kf','Predicted log10kf Error'])
        for index, test_cv in enumerate(Test_CVs):
            print(f'Testing {index} of {len(Test_CVs)}')
            sigma,keq,kf = find_sigma(test_cv)
            filename  = test_cv.replace('.csv','')



            df = pd.read_csv(f'{path_to_test_dir}/{test_cv}',header=None)
            
            if add_noise:
                df.iloc[:,1] = df.iloc[:,1] + noise_level*np.random.randn(df.shape[0])
                df.iloc[:,1] = savgol_filter(x=df.iloc[:,1], window_length=51, polyorder=5)

            range_all = extract(df.copy(),sigma)




            test_feature = range_all.copy()

            #test_feature = scaler.transform(test_feature.reshape(1,-1))
            test_feature = test_feature.reshape(1,-1)


            prediction = model.predict([test_feature,test_feature])
            #print(prediction.shape)
            target = np.log10(np.array([sigma,keq,kf]))



            RFPrediction = RFregressor.predict(test_feature)
            LRPrediction = regressor.predict(test_feature)


            # predict with XGB regressor
            XGBPrediction = np.zeros_like(prediction)
            for i in range(target_train.shape[1]):
                a = XGBregressors[i].predict(test_feature)
                #print(a.shape)
                XGBPrediction[:,i] = a

            
            error = np.average((prediction[0]-target)/target)

            csvwriter.writerow([np.log10(sigma),np.log10(keq),np.log10(kf),prediction[0,1],(pow(10,prediction[0,1])-pow(10,target[1]))/pow(10,target[1]),prediction[0,2],(pow(10,(prediction[0,2]))-pow(10,target[2]))/pow(10,target[2]),(pow(10,(LRPrediction[0,1]))-pow(10,target[1]))/pow(10,target[1]),(pow(10,(LRPrediction[0,2]))-pow(10,target[2]))/pow(10,target[2]),(pow(10,(RFPrediction[0,1]))-pow(10,target[1]))/pow(10,target[1]),(pow(10,(RFPrediction[0,2]))-pow(10,target[2]))/pow(10,target[2]),(pow(10,(XGBPrediction[0,1]))-pow(10,target[1]))/pow(10,target[1]),(pow(10,(XGBPrediction[0,2]))-pow(10,target[2]))/pow(10,target[2])])
            #csvwriter.writerow([sigma,keq,kf,prediction[0,1],(prediction[0,1]-target[1])/target[1],prediction[0,2],(prediction[0,2]-target[2])/target[2]])
            #csvwriter.writerow([sigma,np.log10(keq),np.log10(kf),error,list(range_all)])
            #csvwriter.writerow([sigma,np.log10(keq),np.log10(kf),error,list(target[0])])



            """
            fig = plt.figure(figsize=(16,18))
            ax = plt.subplot(2,1,1)
            plt.scatter(target[0],prediction[0,0],label=r'scan rate, $log_{{10}}V/S$')
            plt.scatter(target[1],prediction[0,1],label=r'keq, $log_{{10}}M$')
            plt.scatter(target[2],prediction[0,2],label=r'kf, $log_{{10}}s^{{-1}}$')
            plt.plot([-10,10],[-10,10],color='r')
            plt.title('Test Results: Neural Network',fontsize='large')

            ax.legend(loc=0,fontsize='large') 
            ax.tick_params(labelsize='large')
            ax.set_xlabel(r"Target Values in log10 scale$", fontweight = "bold",fontsize='large')
            ax.set_ylabel(r"Predicted Values in log10 scales$", fontweight = "bold",fontsize='large')
            #plt.show()

            ax = plt.subplot(2,1,2)
            RFprediction = RFregressor.predict(test_feature)

            plt.scatter(target[0],prediction[0,0],label=r'scan rate, $log_{{10}}V/S$')
            plt.scatter(target[1],prediction[0,1],label=r'keq, $log_{{10}}M$')
            plt.scatter(target[1],prediction[0,2],label=r'kf, $log_{{10}}s^{{-1}}$')
            plt.plot([-10,10],[-10,10],color='r')

            plt.title('Test Results: Random Forest',fontsize='large')
            ax.legend(loc=0,fontsize='large') 
            ax.tick_params(labelsize='large')
            ax.set_xlabel(r"Predicted Values in log10 scales", fontweight = "bold",fontsize='large')
            ax.set_ylabel(r"Predicted Values in log10 scales", fontweight = "bold",fontsize='large')
            fig.savefig(f'{path_to_dir}/{filename} constants Prediction.png')
            plt.close()
            """
            #plt.show()

    writer.close()


        #Visualization of one test sample
        # = find_csv()








    