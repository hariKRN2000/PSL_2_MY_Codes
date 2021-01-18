# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:39:39 2021

@author: hp1
"""

import matplotlib.pyplot as plt 


# Defining a class to plot 

class MyPlot:
    
    def plotTrain(self,y_train, y_pred_train,model = "LinRegn",inter = True):
        
        plt.figure()
        plt.scatter(y_train,y_pred_train,label = 'Prediction')
        plt.plot(y_train,y_train,label = 'Expected')
        plt.xlabel('Target Variable (Training)')
        plt.ylabel(["Prediction by ",model," Model(Training)"])
        plt.title('Predicted data vs Real data')
        plt.legend()
        if inter == False:
            plt.ylim(bottom = 0)
        plt.xlim(left=0)
        plt.show()
        
    def plotTest(self,y_test,y_pred_test,model = "LinRegn",inter = True):
         
        plt.figure()
        plt.scatter(y_test,y_pred_test,label = 'Prediction')
        plt.plot(y_test,y_test,label = 'Expected')
        plt.xlabel('Target Variable (Test)')
        plt.ylabel(['Prediction by ',model,'Model(Test)'])
        plt.title('Predicted data vs Real data')
        if inter == False:
            plt.ylim(bottom = 0)
        plt.xlim(left=0)
        plt.legend()
        plt.show()