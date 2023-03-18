# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:48:50 2020

Non Linear Regressor

Decision Tree Regression
"""

#Importing Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('C:/Users/101985/Desktop/Machine Learning/Machine Learning A-Z Folder/Part 2 - Regression/Section 6 - Polynomial Regression/P14-Polynomial-Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Fitting dataset into Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor =DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

y_pred=regressor.predict([[6.5]])

#Visuallizing Decision Tree prediction

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Position Vs Salary(Decision Tree)')
plt.xlabel='Position'
plt.ylabel='Salary'
plt.show()

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),0.1,color='blue')
plt.title('Position Vs Salary(Decision Tree)')
plt.xlabel='Position'
plt.ylabel='Salary'
