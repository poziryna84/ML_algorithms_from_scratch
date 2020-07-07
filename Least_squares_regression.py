# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:50:44 2020

@author: pozir
"""
import math
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class ls_regression:
    def __init__(self):
        super().__init__
    
    def train_test_split(self, filename, predictors, y_col, split_ratio):
        self.filename = filename
        self.split_ratio = split_ratio
        self.predictors = predictors
        self.y_col = y_col
        data = pd.read_csv(self.filename)
        selected_cols = self.predictors + [self.y_col]
        data = data[selected_cols]

        test_number = round(len(data)*self.split_ratio)
        indexes = [i for i in data.index]

        test_ind = random.choices(indexes, k = test_number)
        train_ind = [i for i in indexes if i not in test_ind]
    
        self.test = data[data.index.isin(test_ind)]
        self.train = data[data.index.isin(train_ind)]
        
        return self.test, self.train
        
    def mean(self, x):
        mean = sum(x)/len(x)
        
        return mean

    def stdev(self, x):
        aux = [(i - self.mean(x))**2 for i in x]
        std = math.sqrt(sum(aux)/(len(x)-1))
        
        return std

    def cor_coof(self, list_x, list_y):
        mean_x = self.mean(list_x)
        mean_y = self.mean(list_y)
        std_x = self.stdev(list_x)
        std_y = self.stdev(list_y)
        aux = 0
        for i in range(len(list_x)):
            z_x = (list_x[i]- mean_x)/std_x
            z_y = (list_y[i]- mean_y)/std_y
            aux += z_x*z_y
        
        return 1/((len(list_x)-1))*aux
    
    def fit_line(self, x, y):
        self.x = x.iloc[:,0].reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.a = (self.stdev(self.y)/self.stdev(self.x))*self.cor_coof(self.x, self.y)
        print('Coefficients: ', self.a)
        self.b = self.mean(self.y) - self.mean(self.x)*self.a
        print('Intercept: ', self.b)
        
        return self.a, self.b
    
  
    def predict(self, x_t, y_t):
        self.x_t = x_t.iloc[:,0].reset_index(drop=True)
        self.y_t = y_t.reset_index(drop=True)
        mean_y = self.mean(self.y_t)
        self.y_preds = []
        error_list = []
        square_error_list = []
        error_from_mean = []
        for i in range(len(self.x_t)):
            pred = self.x_t[i]*self.a + self.b
            self.y_preds.append(pred)
            error_list.append(abs(self.y_t[i] - pred))
            square_error_list.append((self.y_t[i] - pred)**2)
            error_from_mean.append((self.y_t[i] - mean_y)**2)
    
        MAE = self.mean(error_list)
        print('MAE: ', MAE)
        MSE = self.mean(square_error_list)
        print('MSE: ', MSE)
        RMSE = math.sqrt(MSE)
        print('RMSE: ', RMSE)
        R_SQR = (sum(error_from_mean) - sum(square_error_list))/sum(error_from_mean)
        print('R Squared: ', R_SQR)
       
        return self.y_preds

lr = ls_regression()
train_, test_ = lr.train_test_split('data/datasets_33080_43333_car data.csv', ['Kms_Driven'], 'Present_Price', 0.4)
lr .fit_line(train_[['Kms_Driven']], train_['Present_Price'])
preds = lr.predict(test_[['Kms_Driven']], test_['Present_Price'])
# sklearn predictions:
regressor = LinearRegression()
regressor.fit(train_[['Kms_Driven']], train_['Present_Price'])
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(test_[['Kms_Driven']])
print('Mean Absolute Error:', metrics.mean_absolute_error(test_['Present_Price'], y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(test_[['Present_Price']], y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_[['Present_Price']], y_pred)))
print('R Square:', metrics.r2_score(test_[['Present_Price']], y_pred))