# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:11:32 2021

@author: hp1
"""

import numpy as np 
from MyPlot import MyPlot

# Defining my Class for Linear Regression using Gradient Descent
class LinRegGradDesc(MyPlot) :
    def __init__(self,fit_intercept = True)  :
        self.theta = 0
        self.intercept = 0
        self.fit_intercept = fit_intercept
        self.sse = 0
        self.sst = 0
        self.J = 0
    
    
    
    def fit(self,X,y,theta,alpha,num_iter,err_tol = 1e-3) :
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        # add bias if fit_intercept is True
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        m = X.shape[0]  # number of training examples
        for iter in range(int(num_iter)):
            cost = self.compCost(X,y,theta) 
            if cost <= err_tol:
                break
            delta =  np.dot(np.transpose(np.dot(X,theta)-y),X)
            temp = theta - (alpha/m)*np.transpose(delta)
            temp,theta = theta,temp 
        if self.fit_intercept:
            self.intercept = theta[0]
            self.theta = theta[1:]
        else:
            self.intercept = 0
            self.theta = theta
        return (self.intercept,self.theta)
    
    def compCost(self,X, y, theta):
        m = X.shape[0]  # number of training examples
        predictions = np.dot(X,theta)
        sqrErrors = (predictions - y)**2
        self.J = (1/(2*m))*sum(sqrErrors)
        return self.J  
    
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