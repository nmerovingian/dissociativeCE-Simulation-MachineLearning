from re import X
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.signal import savgol_filter 
import matplotlib.pyplot as plt
import time
import os
import csv
import random
import math
from torch.nn import parameter
random.seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.dataset import TensorDataset, random_split


from helper import find_csv,find_sigma
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


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def read_feature(file_name = 'features.csv'):
    data = pd.read_csv(file_name)
    data = data.fillna(0.0)


    targets = data[['Log10scanRate','Log10keq','Log10kf']]
    features = data.drop(['Log10scanRate','Log10keq','Log10kf'],axis=1)



    return features.to_numpy(),targets.to_numpy()


def load_data(features,targets):
    feature_tensor = torch.from_numpy(features)
    target_tensor = torch.from_numpy(targets)

    dataset = TensorDataset(feature_tensor,target_tensor)

    train_dataset, val_dataset = random_split(dataset,[int(len(dataset)*0.8),len(dataset)- int(len(dataset)*0.8)])

    train_loader  = DataLoader(train_dataset,batch_size=24,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=16,shuffle=True)

    return train_loader,val_loader



# Multiheaded DNN
class model(nn.Module):
    def __init__(self,input_shape=None,output_shape=None):
        super().__init__()

        self.d1 = nn.Linear(input_shape,256)
        self.d2 = nn.Linear(256,128)
        self.d3 = nn.Linear(128,64)
        self.d4 = nn.Linear(64,32)

        self.d5 = nn.Linear(input_shape,256)
        self.d6 = nn.Linear(256,128)
        self.d7 = nn.Linear(128,64)
        self.d8 = nn.Linear(64,32)

        self.out = nn.Linear(64,3)

        self.history = {'train_loss':list(),'val_loss':list()}

    def forward(self,x):
        x1 = F.relu(self.d1(x))
        x1 = F.relu(self.d2(x1))
        x1 = F.relu(self.d3(x1))
        x1 = F.relu(self.d4(x1))


        x2 = F.relu(self.d5(x))
        x2 = F.relu(self.d6(x2))
        x2 = F.relu(self.d7(x2))
        x2 = F.relu(self.d8(x2))

        x = torch.cat((x1,x2),dim=1)

        x = self.out(x)
        return x

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

    train_loader, val_loader = load_data(*read_feature())
    model = model(400,3).to(device=device)
    model.double()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss_func = torch.nn.L1Loss()

    for epoch in range(100):

        train_losses = []
        val_losses = []
        for x_batch,y_batch in train_loader:
            model.train()
            x_batch.to(device)
            y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch)

            loss = loss_func(y_hat,y_batch)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            with torch.no_grad():

                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    model.eval()

                    y_hat = model(x_val)

                    val_loss = loss_func(y_val,y_hat)
                    
                    val_losses.append(val_loss.item())


        model.history['train_loss'].append(sum(train_losses)/len(train_losses))
        model.history['val_loss'].append(sum(val_losses)/len(val_losses))
    
    df = pd.DataFrame(model.history)

    df.plot()

    plt.show()












    