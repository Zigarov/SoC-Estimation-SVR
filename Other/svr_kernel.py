"""
In this script, the performances of two different svr were testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ACQUISIZIONE E PREPROCESSING DEI DATI:
data =  pd.read_csv("C:/Users/Acer/Ingegneria/TESI/Batteria/Dati/data.csv") #data: [time,charge,voltage,current]
dataInput = data.iloc[:,[0,2,3]]            #Input: [time,voltage,current]
dataOutput =  data.iloc[:,1]                #Output: [charge]

 #SPLIT:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataInput, dataOutput, test_size=0.3, random_state=0,shuffle = False)

 #SCALING:
from sklearn import preprocessing
x_scaler = preprocessing.MinMaxScaler().fit(X_train)

X_train = x_scaler.transform(X_train)
X_test = x_scaler.transform(X_test)

#REGRESSION:
from sklearn.svm import SVR
svr = [SVR(kernel='rbf', C=100, gamma=0.125),SVR(kernel='linear', C=10)]
y_rbf = svr[0].fit(X_train,y_train).predict(X_test)
y_linear = svr[1].fit(X_train,y_train).predict(X_test)

#EVALUATION:
from sklearn.metrics import r2_score

score_rbf = round(r2_score(y_test, y_rbf),3)
score_linear =  round(r2_score(y_test, y_linear),3)
print('R2 Score for rbf kernel:')
print(score_rbf)
print()
print('R2 Score for linear kernel:')
print(score_linear)


#PLOT SOC(t):
# t = (data.iloc[-len(y_test):,0] - data.iloc[-len(y_test),0])/3600
# plt.plot(t, y_test, label = 'Original', color = 'r')
# plt.plot(t, y_rbf, label = 'Predicted [C=100, kernel = rbf, gamma = 0.125]', color = 'b')
# plt.plot(t, y_linear, label = 'Predicted[C=10, kernel = linear ]', color = 'y')
#
#
# plt.ylabel('SoC (%)')
# plt.xlabel('Time (Hour)')
# plt.title('SoC(t)')
# plt.legend()
# plt.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.99,top = 0.95)
# plt.show()
