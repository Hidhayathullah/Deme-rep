# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset =pd.read_csv('C:/Users/101985/Desktop/Machine Learning/Machine Learning A-Z Folder/Part 1 - Data Preprocessing/Data.csv')
X=dataset.iloc[:,:3].values
Y=dataset.iloc[:,3].values

#Taking care of missing datas
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN',strategy ='mean',axis=0)
imputer =imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding categorical datas

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder =OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Splitting data set into The Training set and the Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Feature Scalling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
