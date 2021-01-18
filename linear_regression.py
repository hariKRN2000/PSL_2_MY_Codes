# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:28:22 2021

@author: hp1
"""
# Import all the needed libraries and modules here
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as ss

 

# Defining my Class for Linear Regression
class LinReg() :
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
        return self.intercept + np.dot(X, self.theta)
    def goodness(self,X,y):
        sse = np.sum((y - self.predict(X))**2)
        sst = np.sum((y - np.mean(y))**2)
        return (1 - sse/sst)
    
# Creating the Training data set :
beta = np.array([[1],[3],[6]])   # This is the vector of all coefficients
inter = 50                      # Set intercept if any
n = 20                          # Number of data points 
x = np.linspace(0,100,n)
x_dat = np.c_[x,x**0.3,x**0.2]
# x = n * ss.uniform.rvs(size=n)         # Uncomment if u want random x
y_dat = inter + np.dot(x_dat,beta) + ss.norm.rvs(loc=0, scale = 2, size = n).reshape(-1,1)     
y_curve = inter + np.dot(x_dat,beta)
plt.figure()
plt.scatter(x,y_dat)
plt.plot(x,y_curve)
plt.xlim(left=0)
plt.title("The random data generated :")

# Fitting the Model 
model = LinReg()
r = 0.6                          # Fraction of data that is training data
X_train = np.c_[x[:int(n*0.6)],x[:int(n*0.6)]**0.3,x[:int(n*0.6)]**0.2]
Y_train = y_dat[:int(n*0.6)]
(Int1,Theta1) = model.fit(X_train,Y_train)
Pred_1 = model.predict(X_train)
Pred_2 = model.predict(x_dat)
R2 = model.goodness(x_dat,y_dat)
print("The parameters obtained from training set : ","Intercept = ",Int1," Coefficients = ",Theta1 ) 
print("The R^2 value is : " ,R2)

# Plotting The fitted curves
plt.figure()
plt.scatter(x[:int(r*n)],y_dat[:int(r*n)])
plt.plot(x[:int(r*n)],Pred_1)
plt.xlim(left=0)
plt.title("The prediction on the Training Set :")
plt.figure()
plt.scatter(x,y_dat)
plt.plot(x,Pred_2)
plt.xlim(left=0)
plt.title("The prediction on the Complete data set :")
