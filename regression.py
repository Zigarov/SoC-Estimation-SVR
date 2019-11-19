"""
In this code, the SVR was modeling by the following steps:
 - Pre-processing: The data was transformed for the machine
 - Cross-Validation: Searching of the best parameters for the SVR
 - Evaluation: Valuation of the SVR Precision
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DATA PRE-PROCESSING:
data =  pd.read_csv("C:/data.csv") #data: [time,charge,voltage,current]

"""
For experiment the dataInput variaton, rebuild the script deleting # in  different dataInput declaration.
"""
dataInput = data.iloc[:,[0,2,3]]                    #Input: [time,voltage,current]
#dataInput = data.iloc[:,2].values.reshape(-1,1)    #Input: [voltage]
#dataInput = data.iloc[:,[0,2]]                     #Input: [time,voltage]
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

#CROSS-VALIDATION:
param_grid = [
  {'C': [1, 10, 100], 'gamma': [0.125, 0.25, 0.5], 'kernel': ['rbf']},
  {'C': [1, 10, 100], 'kernel': ['linear']}
 ]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

svr = GridSearchCV(SVR(), param_grid, cv=5, scoring = 'r2', refit = True, iid = True)
svr.fit(X_train, y_train)

#REGRESSION & VALUATION:
from sklearn.metrics import r2_score

y_pred = svr.predict(X_test)
score = round(r2_score(y_test, y_pred),3)

#PRINT:
 #best parameters
print("Best parameters set found on development set:",svr.best_params_)
print()
 #Grid Scores:
print("Grid scores on development set:")
print()
means = svr.cv_results_['mean_test_score']
stds = svr.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svr.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
print()
 #Test Score:
print("test score: ",score)
print(score)
print()

 #PLOT TABLE:
  #table parameters:
means = np.around(np.asarray(svr.cv_results_['mean_test_score']), decimals=3).reshape(-1,1)
stds = np.around(np.asarray(svr.cv_results_['std_test_score']),decimals=3).reshape(-1,1)
cells = np.append(means,stds, axis=1)
columns = ('mean R2_score', ' std var')
rows =  np.asarray(svr.cv_results_['params'])
colors = [['w','w']]*len(rows)
rowColors = ['w']*len(rows)
colors[svr.best_index_] = ['g','g']
rowColors[svr.best_index_] = 'g'
colors[-2] =  ['g','g']
rowColors[-2] =  'g'

  #figure settings:
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

 #create table:
ax.table(cellText= cells, rowLabels=rows,  colLabels=columns, loc="upper center",cellLoc = 'center', cellColours = colors, rowColours = rowColors)
fig.subplots_adjust(left = 0.26,right = 0.99,bottom = 0,top = 0.99)
plt.show()
