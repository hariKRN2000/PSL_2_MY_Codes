# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 21:04:38 2021

@author: hp1
"""
# Importing libraries/modules
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from LinReg import LinReg  # My model!
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from LinRegGradDesc import LinRegGradDesc  # My model !



# Importing data:
data_m = pd.read_csv('car data.csv')
print('Imported data is : ')
print(data_m.head())
print('The Features of this data set are: ')
print(data_m.info())

# Viewing the data
#print(data_m.head()) # Prints only first 5 entries

# Some Pre-Processing :
# Adding age of the car instead of year brought
data_m['Age'] = 2020 - data_m['Year']
data_m.drop(labels = 'Year', axis = 1, inplace = True)
# Removing car name as it doesnt have any use in prediction
data_m.drop(labels = 'Car_Name',axis = 1, inplace = True)
# Creating dummies for Catagorical features : 
data_m = pd.get_dummies(data = data_m,drop_first=True)

# Creating testing and training data
# Obtaining target variable and its features
y = data_m['Selling_Price']
X = data_m.drop('Selling_Price',axis=1)
print('The modified data set is :')
print(data_m.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Fitting Curve with the Linear Regresion Nodel of Sci-kit learn 
model1 =  LinearRegression()
model1.fit(X_train,y_train)   # Training the model
# Obtaining R2 value of Training set
y_pred_train_1= model1.predict(X_train)
R2_train_model_1 = r2_score(y_train,y_pred_train_1)
# Obtaining R2 value of Test set
y_pred_test_1= model1.predict(X_test)
R2_test_model_1 = r2_score(y_test,y_pred_test_1)

# Fitting Curve with the Linear Regresion tool made by me (With Intercept)
model2 =  LinReg()
(Int1,Theta1) = model2.fit(X_train,y_train)     # Training the model
# Obtaining R2 value of Training set
y_pred_train_2= model2.predict(X_train) 
R2_train_model_2 = model2.goodness(X_train,y_train)
# Obtaining R2 value of Test set
y_pred_test_2= model2.predict(X_test)
R2_test_model_2 = model2.goodness(X_test,y_test)

# Fitting a curve using a Random Forest Regressor
model3 = RandomForestRegressor()
model3.fit(X_train,y_train)   # Training the model
# Obtaining R2 value of Training set
y_pred_train_3= model3.predict(X_train)
R2_train_model_3 = r2_score(y_train,y_pred_train_3)
# Obtaining R2 value of Test set
y_pred_test_3= model3.predict(X_test)
R2_test_model_3 = r2_score(y_test,y_pred_test_3)

# Fitting Curve with the Linear Regresion tool made by me ( Without Intercept)
model4 =  LinReg(fit_intercept = False)
(Int2,Theta2) = model4.fit(X_train,y_train)     # Training the model
# Obtaining R2 value of Training set
y_pred_train_4= model4.predict(X_train) 
R2_train_model_4 = model4.goodness(X_train,y_train)
# Obtaining R2 value of Test set
y_pred_test_4= model4.predict(X_test)
R2_test_model_4 = model4.goodness(X_test,y_test)

# Fitting Curve with Gradient Descent tool made by me (With Intercept)
model5 = LinRegGradDesc()
theta0 = np.array([5.421,4.37e-1,-5.3e-6,3.4e-1,-4.1e-1,2.2,4.5e-1,-1.2,-1.87])  # Initial Guess for theta
alpha = 1e-10                    # Learning Rate for gradient descemt
num_iter = 10000               # number of iterations for GD
(Int3,Theta3) = model5.fit(X_train,y_train,theta0,alpha,num_iter)    # Training the model         
# Obtaining R2 value of Training set
y_pred_train_5= model5.predict(X_train) 
R2_train_model_5 = model5.goodness(X_train,y_train)
# Obtaining R2 value of Test set
y_pred_test_5= model5.predict(X_test)
R2_test_model_5 = model5.goodness(X_test,y_test)

# Fitting Curve with Gradient Descent tool made by me (Without Intercept)
model6 = LinRegGradDesc(fit_intercept = False)
theta0 =  np.array([4.42894370e-01, -5.29889993e-06,  3.04699660e-01, -3.93630346e-01,
        7.25299803e+00,  5.49822070e+00, -1.16856809e+00, -1.63920545e+00])  # Initial Guess for theta
alpha = 1e-10                    # Learning Rate for gradient descemt
num_iter = 10000               # number of iterations for GD
(Int4,Theta4) = model6.fit(X_train,y_train,theta0,alpha,num_iter)    # Training the model         
# Obtaining R2 value of Training set
y_pred_train_6= model6.predict(X_train) 
R2_train_model_6 = model6.goodness(X_train,y_train)
# Obtaining R2 value of Test set
y_pred_test_6= model6.predict(X_test)
R2_test_model_6 = model6.goodness(X_test,y_test)

# Plotting Results : 
# Training by scikit Model
plt.figure()
plt.scatter(y_train,y_pred_train_1,label = 'Prediction')
plt.plot(y_train,y_train,label = 'Expected')
plt.xlabel('Target Variable (Training)')
plt.ylabel('Prediction by Sci-Kit Model(Training)')
plt.title('Predicted data vs Real data')
plt.xlim(left=0)
plt.legend()
plt.show()

# Test by scikit Model
plt.figure()
plt.scatter(y_test,y_pred_test_1,label = 'Prediction')
plt.plot(y_test,y_test,label = 'Expected')
plt.xlabel('Target Variable (Test)')
plt.ylabel('Prediction by Sci_kit Model(Test)')
plt.title('Predicted data vs Real data')
plt.xlim(left=0)
plt.legend()
plt.show() 

# Training and Testing of My Model (with intercept) : 
model2.plotTrain(y_train,y_pred_train_2,model = "My Model (with inter)")  # Plotting Training set
model2.plotTest(y_test,y_pred_test_2,model = "My Model (with inter)")     # Plotting Test set 

# Training by Random Forest Model
plt.figure()
plt.scatter(y_train,y_pred_train_3,label = 'Prediction')
plt.plot(y_train,y_train,label = 'Expected')
plt.xlabel('Target Variable (Training)')
plt.ylabel('Prediction by Random Forest Model(Training)')
plt.title('Predicted data vs Real data')
plt.xlim(left=0)
plt.legend()
plt.show()

# Test by Random Forest Model
plt.figure()
plt.scatter(y_test,y_pred_test_3,label = 'Prediction')
plt.plot(y_test,y_test,label = 'Expected')
plt.xlabel('Target Variable (Test)')
plt.ylabel('Prediction by Random Forest Model(Test)')
plt.title('Predicted data vs Real data')
plt.xlim(left=0)
plt.legend()
plt.show()

# Training and Testing of My Model (without intercept) : 
model4.plotTrain(y_train,y_pred_train_4,model = "My Model (without inter)",inter=False)  # Plotting Training set
model4.plotTest(y_test,y_pred_test_4,model = "My Model (without inter)",inter=False)     # Plotting Test set 

# Training and Testing of My GD Model (with intercept) : 
model5.plotTrain(y_train,y_pred_train_5,model = "My GD Model (with inter)")  # Plotting Training set
model5.plotTest(y_test,y_pred_test_5,model = "My GD Model (with inter)")     # Plotting Test set

# Training and Testing of My GD Model (with intercept) : 
model6.plotTrain(y_train,y_pred_train_6,model = "My GD Model (without inter)")  # Plotting Training set
model6.plotTest(y_test,y_pred_test_6,model = "My GD Model (without inter)")     # Plotting Test set 

# Printing data 
print_data = {}
print_data['Model'] = ['Sci-Kit Learn Model', 'My LinReg Model','Random Forest Model',
                       'My LinReg Model without intercept','My GD Model',
                        'My GD Model without intercept']
                         
print_data['R^2 Value (Training)'] = [R2_train_model_1,R2_train_model_2,R2_train_model_3,
                                    R2_train_model_4,R2_train_model_5,R2_train_model_6]
                          
                          
                          
print_data['R^2 Value (Testing)'] = [R2_test_model_1,R2_test_model_2,R2_test_model_3,
                                    R2_test_model_4,R2_test_model_5,R2_test_model_6]
show_data = pd.DataFrame(data = print_data)
print(show_data)



















