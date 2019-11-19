"""
In this code, the raw data was extracted from a CSV file to a pandas dataframe.
After the printing of the main info, from the original data was selected a shorter sampling interval
but whit a pseudo-costant sampling frequencies.
After that, the new data was filtered for noise elimination.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#DATAINFO:
rawData = "C:/rawData.csv"
data =  pd.read_csv(rawData)
dt = np.diff(data['time'])          # out[i] = a[i+1] - a[i], i =0,..,N-2
start = int(data.loc[0,'time'])
end = int(data.iloc[-1,0])
samples = len(data['time'])

 #print dataInfo:
print()
print("The series started in: ",datetime.fromtimestamp(start))
print("And ended in: ", datetime.fromtimestamp(end))
print("Inizial number of samples: ", samples)
print()



#DATA ELABORATION: (SUBSET)
t = None                    #index of too big dt
t_ = None                   #index of negative dt

for i in range(0,len(dt)):
    if (dt[i]<0):
        t_ = np.append(t_,i+1)
    if (dt[i]>1800):
        t = np.append(t,i)

t = t[1:]
t[0]=t[0]+1
 #data subset:
data = data.iloc[t[0]:t[1]]
t_ = t_[1:]
for i in range(0,len(t_)):
    if ((t[2]-t[1])>t_[i]-t[1]>=0):
        data = data.drop(t_[i])
data.reset_index()

 #new data info:
start = data.iloc[0,0]
end = data.iloc[-1,0]
samples = len(data['time'])
print()
print("The new series started in: ",datetime.fromtimestamp(start))
print("And ended in: ", datetime.fromtimestamp(end))
print("New number of samples: ", samples)
print()
 #Save new data:
data.to_csv("C:/newData.csv",index = False)

#FILTERING DATA:
indexData = None;

for i in range(0, samples):
    q, r = divmod(i, 60)
    if r is 0:
        indexData = np.append(indexData,i)
indexData = indexData[1:]

data = data.iloc[indexData].reset_index()


 #filtered data info:
samples = len(data['time'])
print()
print("New number of filtered samples: ", samples)
print()

 #saving filtered data:
data.to_csv("C:/data.csv",index = False)
