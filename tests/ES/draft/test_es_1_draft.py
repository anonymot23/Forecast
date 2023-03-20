import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# simple test : 30 min
## create project more/less clean containing multiple model 
### folder for data  ok
### file for data preparation  ok 
### file for model ok 
### eventually file for utils 
### file to generate forecast ok
### main file combining everything ok
### file for global parameters ok 
### folder for testing ok 


# parameters
lag = 12

# collect and load data 
data_folder = "../../data"
filename = 'AirPassengers.csv'
dataAir = pd.read_csv(f"{data_folder}/{filename}")


# prepare/pre-process data 
dataAir['Month'] = pd.to_datetime(dataAir['Month'])
dataAir.set_index(dataAir['Month'], inplace=True)
scaler = StandardScaler()
passengers = dataAir['#Passengers'].values.reshape((-1, 1))
scaler.fit(passengers)
dataAir['#Passengers'] = scaler.transform(passengers)

## split train/test data 
timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]

dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
indexTrain = dataTrain.index
valTrain = dataTrain['#Passengers'].values
yTrain = pd.Series(valTrain, indexTrain)

dataTest = dataAir[timeTest[0]: timeTest[1]]
indexTest = dataTest.index
valTest = dataTest['#Passengers'].values
yTest = pd.Series(valTest, indexTest)

# build and fit model 
minValTrain = valTrain.min()-0.1
modelES = ExponentialSmoothing(yTrain-minValTrain, 
                               seasonal_periods=lag, 
                               trend='add',
                               seasonal='add',
                               use_boxcox=True,
                               initialization_method='estimated')
modelESFitted = modelES.fit()

# test/evaluate the model  
lenTest = len(yTest) 
predTest = modelESFitted.forecast(lenTest) + minValTrain
error = mean_squared_error(predTest, yTest)
print(error)

# plot values 
plt.plot(predTest)
plt.plot(yTest)
plt.show()