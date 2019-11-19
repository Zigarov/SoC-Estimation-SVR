import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DATA PROCESSING:
data =  pd.read_csv("C:/Users/Acer/Ingegneria/TESI/Batteria/Dati/data.csv") #data: [time,charge,voltage,current]

dataInput = data.iloc[:,[0,2,3]]                   #Input: [time,voltage,current]
#dataInput = data.iloc[:,2].values.reshape(-1,1)    #Input: [voltage]
#dataInput = data.iloc[:,[0,2]]                      #Input: [time,voltage]
dataOutput =  data.iloc[:,1]                        #Output: [charge]

 #split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataInput, dataOutput, test_size=0.3, random_state=0,shuffle = False)

 #scaling:
from sklearn import preprocessing
x_scaler = preprocessing.MinMaxScaler().fit(X_train)
y_scaler = preprocessing.MinMaxScaler()

X_train = x_scaler.transform(X_train)
X_test = x_scaler.transform(X_test)

#REGRESSION & VALUATION:
from sklearn.svm import SVR

svr = SVR(kernel = 'linear', C = 10)
svr.fit(X_train, y_train)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

y_pred = svr.predict(X_test)
r2 = round(r2_score(y_test, y_pred),3)
absolute = round(mean_absolute_error(y_test, y_pred),3)
squared = round(mean_squared_error(y_test, y_pred),3)

#PRINT SCORES:
print()
print("r2_score:")
print(r2)
print()
print("mean_absolute_error:")
print(absolute)
print()
print("mean_squared_error")
print(squared)
print()

 #PLOT SOC(t):
# plt.plot((data.iloc[-len(y_pred):,0] - data.iloc[-len(y_pred),0])/3600,y_pred, label = 'Predicted', color = 'b')
# plt.plot((data.iloc[-len(y_pred):,0] - data.iloc[-len(y_pred),0])/3600,y_test, label = 'Original', color = 'r')
# plt.ylabel('SoC (%)')
# plt.xlabel('Time (Hour)')
# plt.title('SoC(t)')
# plt.legend()


 #PLOT SOC(V):
# plt.scatter(data.iloc[-len(y_pred):,2], y_pred, label = 'Predicted', color = 'b')
# plt.scatter(data.iloc[-len(y_pred):,2],y_test, label = 'Original', color = 'r')
# plt.title('SoC(V)')
# plt.xlabel('Voltage (mV)')
# plt.ylabel('SoC (%)')
# plt.legend()
# plt.show()
