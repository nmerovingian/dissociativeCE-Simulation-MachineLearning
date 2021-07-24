import getpass
from keras import activations
from keras.engine.input_layer import Input
from keras.engine.training import Model
if getpass.getuser()=='RGCGroup':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #Run with CPU only
import pandas as pd
import numpy as np
from keras.layers import Dense,BatchNormalization,Dropout,Input
from keras.layers.merge import concatenate
from keras.models import Sequential,Model
from keras.layers import LeakyReLU
from keras.callbacks import LearningRateScheduler
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
import json
import ast
from helper import find_csv,find_sigma
from ann_visualizer.visualize import ann_viz
import csv
linewidth = 3
fontsize = 14
figsize = [10,8]
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


epochs=3000
path_to_dir = './Test Data'  # Path to directory with testing data 
def read_feature(file_name = 'features.csv'):
    data = pd.read_csv(file_name)
    data = data.fillna(0.0)


    features = data[['Log10scanRate','Log10keq','Log10kf']]
    targets = data.drop(['Log10scanRate','Log10keq','Log10kf'],axis=1)

    return features,targets

def create_model_potential(data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
    model = Sequential()
    model.add(Dense(512,input_dim =data.shape[1] ,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(2048,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(output_shape,activation='linear'))
    model.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error','mean_absolute_error'])
    return model


def create_model_current(data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
    model = Sequential()
    model.add(Dense(512,input_dim =data.shape[1] ,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(2048,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(output_shape,activation='linear'))
    model.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error','mean_absolute_error'])
    return model


class DualNetwork(object):
    def __init__(self,data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
        output_shape = int(output_shape/2)
        self.model_potential = create_model_potential(data,output_shape,optimizer,loss)
        self.model_current = create_model_current(data,output_shape,optimizer,loss)


    def fit(self,data_train,target_train,epochs,batch_size=32,validation_split=0.2,verbose=1,**kwargs):

        target_train_potential  = pd.concat((target_train.iloc[:,0:100],target_train.iloc[:,200:300]),axis=1)
        target_train_current  = pd.concat((target_train.iloc[:,100:200],target_train.iloc[:,300:400]),axis=1)
        self.model_potential.fit(data_train,target_train_potential,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=verbose,**kwargs)
        self.model_current.fit(data_train,target_train_current,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=verbose,**kwargs)

        return (self.model_potential.history,self.model_current.history)

    def predict(self,data_prediction):
        potential_prediction = self.model_potential.predict(data_prediction)
        current_prediction = self.model_current.predict(data_prediction)

        return np.hstack((potential_prediction[:,0:100],current_prediction[:,0:100],potential_prediction[:,100:200],current_prediction[:,100:200]))

    def save_weights(self):
        self.model_potential.save_weights('./weights/model_potential.h5')
        self.model_current.save_weights('./weights/model_current.h5')

    def load_weights(self):
        self.model_potential.load_weights('./weights/model_potential.h5')
        self.model_current.load_weights('./weights/model_current.h5')

    def weight_exists(self):
        if os.path.exists('./weights/model_potential.h5'):
            return True
        else:
            return False

    def plot_model(self):
        plot_model(self.model_potential,to_file='Predict Voltammogram Potentials.png',show_shapes=True,show_layer_names=True,dpi=400)
        plot_model(self.model_current,to_file='Predict Voltammogram Currents.png',show_shapes=True,show_layer_names=True,dpi=400)
        ann_viz(self.model_current,filename='Predict Voltammogram Potential.gv',title='Predicting Fluxes of Voltammograms',view=False)
        ann_viz(self.model_potential,filename='Predict Voltammogram Current.gv',title='Predicting Potentials of Voltammograms',view=False)


"""
def create_model(data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
    # Head 1
    inputs1 = Input(shape=(data.shape[1],))
    dnn11 = Dense(512,activation='relu')(inputs1)
    dnn12 = Dense(1024,activation='relu')(dnn11)
    dnn13 = Dense(2048,activation='relu')(dnn12)
    dnn14 = Dense(4096,activation='relu')(dnn13)
    outputs1 = Dense(output_shape-2)(dnn14)

    #head 2
    inputs2 = Input(shape=(data.shape[1],))
    dnn21 = Dense(256)(inputs2)
    dnn22 = Dense(128)(dnn21)
    dnn23 = Dense(64)(dnn22)
    outputs2 = Dense(2)(dnn23)

    outputs = concatenate([outputs1,outputs2])

    model = Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error','mean_absolute_error'])

    return model
"""
    

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



# Schedule for learning rate decay
def schedule(epoch,lr):
    if epoch <1000:
        return lr
    else:
        return lr*0.9999



if __name__ == "__main__":
    start = time.perf_counter()

    features,targets = read_feature()
    data_train,data_test,target_train,target_test = train_test_split(features,targets,test_size=0.2)
    print(data_train.shape)

    scheduler = LearningRateScheduler(schedule,verbose=0)
    """
    print(data.dtypes)
    print(target.head())
    print(data.head())
    """
    #No need for scaler
    """
    scaler = StandardScaler()
    scaler = scaler.fit(data_train)

    print(features.shape)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)"""

    
    #Fit with NN
    model = DualNetwork(data=data_train,output_shape=targets.shape[1])
    model.plot_model()


    """

    if not os.path.exists('Predict Voltammogram Potential.h5'):


        model.save_weights('Predict Voltammogram Potential.h5')
        with open('Predict Voltammogram Potential History.json','w') as f:
            json.dump(str(history.history),f)
    else:
        model.load_weights('Predict Voltammogram Potential.h5')

    """
    pred = model.predict(data_test)

    if not model.weight_exists():
        history1,history2 = model.fit(data_train,target_train,epochs=epochs,batch_size=32,validation_split=0.2,verbose=1)
        model.save_weights()
    else:
        model.load_weights()


    print(pred.shape)
    #print('predicted value',pred,'actual value',target_test)


    #Fit with RandomForest
    RFregressor = RandomForestRegressor(n_estimators=20)
    RFregressor.fit(data_train,target_train)
    RFPredicted = RFregressor.predict(data_test)

    #Fit with LinearRegression
    regressor = LinearRegression()
    regressor.fit(data_train,target_train)
    LRPredicted = regressor.predict(data_test)

    #Fit with XGBRegressor
    XGBregressors = dict()
    for i in range(data_train.shape[1]):

        XGBregressors[i] = XGBRegressor()
    
        XGBregressors[i].fit(data_train,target_train.iloc[:,i])
    
    # predict with XGB regressor
    XGBpredicted = np.zeros_like(pred)
    for i in range(data_train.shape[1]):
        a = XGBregressors[i].predict(data_test)
        print(a.shape)
        XGBpredicted[:,i] = a
    


    NNerror = mean_squared_error(target_test,pred)
    RFerror = mean_squared_error(target_test,RFPredicted)
    LRerror = mean_squared_error(target_test,LRPredicted)
    NNABSerror = mean_absolute_error(target_test,pred)
    RFABSerror = mean_absolute_error(target_test,RFPredicted)
    LRABSerror = mean_absolute_error(target_test,LRPredicted)
    XGBerror = mean_squared_error(target_test,XGBpredicted)
    XGBABSerror = mean_absolute_error(target_test,XGBpredicted)


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
    
    fig.savefig('Four  Model Results.png')

    plt.show()

    """
    with open('Predict Voltammogram Potential History.json','r') as f:
        training_history = pd.DataFrame(ast.literal_eval(json.load(f)))
    
    training_history.plot()
    plt.show()
    """

    #visualization of one test sample 


    Test_CVs=find_csv(path_to_dir=path_to_dir)

    prediction_time = np.zeros(len(Test_CVs))

    with open('predict voltammogram potential.csv',newline='',mode='w') as writer:
        csvwriter = csv.writer(writer)
        csvwriter.writerow(['log10scanRate','log10keq','log10kf','forward scan potential error','forward scan current error','reverse scan potential error','reverse scan current error','RFforward scan potential error','RFforward scan current error','RFreverse scan potential error','RFreverse scan current error'])
        for index, test_cv in enumerate(Test_CVs):
            print(f'Testing {index+1} of {len(Test_CVs)} Testing Data ')
            sigma,keq,kf = find_sigma(test_cv)
            filename  = test_cv.replace('.csv','')



            df = pd.read_csv(f'{path_to_dir}/{test_cv}',header=None)
            range_all = extract(df.copy(),sigma)


            test_feature = np.array([[sigma,keq,kf]])
            test_feature = np.log10(test_feature)
            #test_feature = scaler.transform(test_feature)
            time_start = time.perf_counter()
            target = model.predict(test_feature)
            prediction_time[index] = time.perf_counter() - time_start
            
            fig = plt.figure(figsize=(16,18))
            ax = plt.subplot(2,1,1)
            df.plot(x=0,y=1,label=f'test voltammogram, scan rate={sigma:.2E} V/s,keq={keq:.2E}$M$,kf={kf:.2E}$s^{{-1}}$',ax=ax,alpha=0.6)

            plt.scatter(target[0,0:100],target[0,100:200],label='Prediction of Forward Scan')
            plt.scatter(target[0,200:300],target[0,300:400],label='Prediction of Reverse Scan')
            #plt.scatter(range_all[0:100],range_all[100:200],label='Forward Scan',color='k')
            #plt.scatter(range_all[200:300],range_all[300:400],label='Reverse Scan',color='b')
            plt.title('Test Results: Neural Network',fontsize='large')

            ax.legend(loc=0,fontsize='large') 
            ax.tick_params(labelsize='large')
            ax.set_xlabel(r"Potential,$\theta$", fontweight = "bold",fontsize='large')
            ax.set_ylabel(r"Standardozed Flux,$J/sqrt(\sigma)$", fontweight = "bold",fontsize='large')
            #plt.show()

            ax = plt.subplot(2,1,2)
            df.plot(x=0,y=1,label=f'test voltammogram, scan rate={sigma:.2E} V/s,keq={keq:.2E}$M$,kf={kf:.2E}$s^{{-1}}$',ax=ax)
            RFtarget = RFregressor.predict(test_feature)



            plt.scatter(RFtarget[0,0:100],RFtarget[0,100:200],label='Prediction of Forward Scan',color='k')
            plt.scatter(RFtarget[0,200:300],RFtarget[0,300:400],label='Prediction of Reverse Scan',color='b')
            plt.title('Test Results: Random Forest',fontsize='large')
            ax.legend(loc=0,fontsize='large') 
            ax.tick_params(labelsize='large')
            ax.set_xlabel(r"Potential,$\theta$", fontweight = "bold",fontsize='large')
            ax.set_ylabel(r"Standardozed Flux,$J/sqrt(\sigma)$", fontweight = "bold",fontsize='large')


            
            fig.savefig(f'{path_to_dir}/{filename} Prediction Voltammogram.png')



            fig,ax = plt.subplots(figsize=(16,9))
            df.plot(x=0,y=1,label=f'Original Voltammogram',ax=ax,alpha=0.6)

            ax.scatter(target[0,0:100],target[0,100:200],label='Prediction of Forward Scan',alpha=0.8,color='#DB4437')
            ax.scatter(target[0,200:300],target[0,300:400],label='Prediction of Reverse Scan',alpha=0.8,color='#F4B400')
            ax.set_xlabel(r"Potential,$\theta$", fontweight = "bold",fontsize='large')
            ax.set_ylabel(r"Flux,$J$", fontweight = "bold",fontsize='large')

            ax.legend()

            fig.savefig(f'{path_to_dir}/{filename} Method B.png',dpi=400,bbox_inches ='tight')

            plt.close('all')
            
            
            # Save the predicted voltammogram
            potential = np.append(target[0,0:100],target[0,200:300])
            flux = np.append(target[0,100:200],target[0,300:400])
            predicted_voltammogram = np.stack((potential,flux),axis=1)

            predicted_voltammogram = pd.DataFrame(predicted_voltammogram)
            predicted_voltammogram.to_csv(f'forward-reverse/{filename} Prediction.csv',index=False)


            forward_potential_error = np.average((target[0,0:100]-range_all[0:100]))
            forward_current_error = np.average((target[0,100:200]-range_all[100:200]))
            reverse_potential_error = np.average((target[0,200:300]-range_all[200:300]))
            reverse_current_error = np.average((target[0,300:400]-range_all[300:400]))

            RFforward_potential_error = np.average((RFtarget[0,0:100]-range_all[0:100]))
            RFforward_current_error = np.average((RFtarget[0,100:200]-range_all[100:200]))
            RFreverse_potential_error = np.average((RFtarget[0,200:300]-range_all[200:300]))
            RFreverse_current_error = np.average((RFtarget[0,300:400]-range_all[300:400]))
            csvwriter.writerow([np.log10(sigma),np.log10(keq),np.log10(kf),forward_potential_error,forward_current_error,reverse_potential_error,reverse_current_error,RFforward_potential_error,RFforward_current_error,RFreverse_potential_error,RFreverse_current_error])
            
    


    writer.close()


    print(np.average(prediction_time))









    