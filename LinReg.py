# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:27:12 2021

@author: hp1
"""

import numpy as np 
from MyPlot import MyPlot
# Defining my Class for Linear Regression
class LinReg(MyPlot) :
    def __init__(self,fit_intercept = True)  :
        self.theta = 0
        self.intercept = 0
        self.fit_intercept = fit_intercept
        self.sse = 0
        self.sst = 0
        
    def fit(self,X,y) :
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        # add bias if fit_intercept is True
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        # Defining data and target 
        self.data = X 
        self.target = y
        # Using the Normal Equation
        T = np.linalg.inv(np.dot(np.transpose(X),X))
        Tx= np.dot(np.transpose(X),y) 
        theta = np.dot(T,Tx)
        # set attributes
        if self.fit_intercept:
            self.intercept = theta[0]
            self.theta = theta[1:]
        else:
            self.intercept = 0
            self.theta = theta
        return (self.intercept,self.theta)
    def predict(self,X):
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1) 
        self.pred = self.intercept + np.dot(X, self.theta)
        return self.pred
    def goodness(self,X,y):
        sse = np.sum((y - self.predict(X))**2)
        sst = np.sum((y - np.mean(y))**2)
        return (1 - sse/sst)