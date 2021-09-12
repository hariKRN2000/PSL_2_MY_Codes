# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 00:55:38 2021

@author: hari
"""
import numpy as np
import pandas as pd

class Wilson :
    
    
    def __init__(self):
        self.V1 = 52.5
        self.V2 = 76.46
    
    def getParam(self, T, A1, A2 ) :
        R = 8.314462
        self.Lambda12 = (self.V1/self.V2)*np.exp(-A1/R/T)
        self.Lambda21 = (self.V2/self.V1)*np.exp(-A2/R/T)
        return (self.Lambda12,self.Lambda21)
    
    def getGamma(self,T,x_1, A1, A2) :
        x_2 = 1 - x_1 
        (L12,L21) = self.getParam(T,A1,A2)
        ln_gamma1 = -np.log(x_1 + x_2*L12) + x_2*( (L12/(x_1 + x_2*L12)) - (L21/(x_2 + x_1*L21)))
        ln_gamma2 = -np.log(x_2 + x_1*L21) - x_1*( (L12/(x_1 + x_2*L12)) - (L21/(x_2 + x_1*L21)))
        self.gama1 = np.exp(ln_gamma1)
        self.gama2 = np.exp(ln_gamma2)
        return (self.gama1,self.gama2)
    
    def getGe(self,T, x_1) :
        R = 8.314462
        x_2 = 1-x_1
        self.ge_RT = x_1*np.log(self.gama1) + x_2*np.log(self.gama2)
        return self.ge_RT*R*T
    
    def ObjectiveFunc(self,A, x1, T_K,ge ) :
        R = 8.314462
        x1 = x1.to_numpy() ; T_K = T_K.to_numpy() ; ge = ge.to_numpy()
        lambda12 = np.zeros(len(x1)) ; lambda21 = np.zeros(len(x1))
        gam1 = np.zeros(len(x1)) ; gam2 = np.zeros(len(x1))
        Ge = np.zeros(len(x1))
        for i in range(1,len(x1)-1):
            (L12,L21) = self.getParam(T_K[i], A[0], A[1])
            lambda12[i],lambda21[i] = L12,L21
            (g1,g2) = self.getGamma(T_K[i],x1[i], A[0], A[1])
            gam1[i],gam2[i] = g1,g2
            Ge[i] = self.getGe(T_K[i],x1[i])
        
    
        error = np.sum((ge - Ge)**2)   
   
        return error
