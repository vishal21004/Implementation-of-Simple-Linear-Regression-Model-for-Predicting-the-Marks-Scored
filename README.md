# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict regression for marks by representing in a graph.
6. Compare graphs and hence linear regression is obtained for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: VISHAL M.A

RegisterNumber: 212222230177
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## DataSet:
![dataset](https://github.com/user-attachments/assets/292127b3-869a-4cf4-b4c4-74490db1212c)

## Hard Values:
![hrdvlue](https://github.com/user-attachments/assets/7d69e66c-ff9d-440a-8521-183e447a86de)

## Tail Values:
![tl vl](https://github.com/user-attachments/assets/8cc75cde-f301-49bb-bb8d-547383b86314)

## X and Y Values:
![xy vl](https://github.com/user-attachments/assets/a4599a20-787d-4caf-9742-e3e4a35dcbe8)

## Prediction of X and Y:
![pr xy](https://github.com/user-attachments/assets/2e180613-0ff2-4c59-93ce-cd581c3fbd1b)

## MSE, MAE and RMSE:
![mae](https://github.com/user-attachments/assets/74873a51-ec05-441a-9035-b9f1bce5eb08)

## Training Set:
![trnng dtst](https://github.com/user-attachments/assets/4b249163-5b0a-4efa-b864-1db1bd1375cc)
![trng 2](https://github.com/user-attachments/assets/4e48d611-ec67-4841-96e8-d58754407fe3)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
