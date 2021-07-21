import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import os


def get_apparent_transfer_coefficient(CV,start=0.1,end=0.3):
    df = pd.read_csv(CV,header=None)
    cv_forward = df[:int(len(df)/2)]
    min_current = cv_forward[1].min()
    min_current_index = cv_forward[1].idxmin()

    cv_before_peak = cv_forward.iloc[:min_current_index]

    cv_tafel = cv_before_peak[(cv_before_peak[1]<start*min_current)&(cv_before_peak[1]>end*min_current)]

    cv_tafel[1] = np.log(-cv_tafel[1])



    model = LinearRegression()
    x = pd.DataFrame(cv_tafel[0])
    y = pd.DataFrame(cv_tafel[1])
    model.fit(x,y)
    return model.coef_[0][0]



if __name__ == "__main__":
    get_apparent_transfer_coefficient('var1=38.9434var2=0var3=0.csv')