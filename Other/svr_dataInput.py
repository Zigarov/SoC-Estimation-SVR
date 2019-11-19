"""
In this script, the performance of the two best models founded was tested by changing the dataInput set.
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

#REGRESSION
from sklearn.svm import SVR
#(delete # for different svr parameters):
#svr = SVR(kernel='rbf', C=100, gamma=0.125)
svr = SVR(kernel='linear', C=10,)
 #from [V]:
svr.fit(X_train[:,1].reshape(-1,1), y_train),
y_pred0 = svr.predict(X_test[:,1].reshape(-1,1))
 #from [v,i]:
svr.fit(X_train[:,[0,1]], y_train),
y_pred1 = svr.predict(X_test[:,[0,1]])
 #from [t,V,I]:
svr.fit(X_train, y_train)
y_pred2 = svr.predict(X_test)

#PLOT SOC(t):
# t = (data.iloc[-len(y_test):,0] - data.iloc[-len(y_test),0])/3600
# plt.plot(t, y_pred0, label = 'Predicted from [V]', color = 'b')
# plt.plot(t, y_pred1, label = 'Predicted from [V,I]', color = 'y')
# plt.plot(t, y_pred2, label = 'Predicted from [t,V,I]', color = 'g')
# plt.plot(t, y_test, label = 'Original', color = 'r')
#
# plt.ylabel('SoC (%)')
# plt.xlabel('Time (Hour)')
# #plt.title('SoC(t) (C=100, kernel = rbf, gamma = 0.125)')
# plt.title('SoC(t) (C=10, kernel = linear')
# plt.legend()
# plt.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.99,top = 0.95)
# plt.show()

#PLOT TABLE:
 #table parameters:
columns = ('mean R2_score', ' std var')
rows = ('V','[V,I]','[t,V,I]')
rbf = [[0.993, 0.04],[0.993, 0.04],[0.996,0.05]]
linear = [[0.992, 0.05],[0.993, 0.05],[0.996,0.03]]
# colors = [['w','w']]*len(rows)
# rowColors = ['w']*len(rows)
# colors[svr.best_index_] = ['g','g']
# rowColors[svr.best_index_] = 'g'
# colors[-2] =  ['g','g']
# rowColors[-2] =  'g'
 #figure settings:
fig, axs = plt.subplots(1,2)
fig.patch.set_visible(False)
axs[0].axis('off')
axs[0].axis('tight')
axs[0].set_title('C=100, kernel = rbf, gamma = 0.125')
axs[1].set_title('C=10, kernel = linear')
axs[1].axis('off')
axs[1].axis('tight')

 #create table:
axs[0].table(cellText= rbf, rowLabels=rows,  colLabels=columns, loc="upper center", cellLoc = 'center')
axs[1].table(cellText= linear,  colLabels=columns, loc="upper center", cellLoc = 'center')
fig.subplots_adjust(left = 0.26,right = 0.99,bottom = 0,top = 0.99)

plt.show()
