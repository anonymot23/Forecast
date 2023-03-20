import pandas as pd
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
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
data_folder = "../../../data"
filename = 'AirPassengers.csv'
dataAir = pd.read_csv(f"{data_folder}/{filename}")


# prepare/pre-process data 
dataAir['Month'] = pd.to_datetime(dataAir['Month'])
dataAir.set_index(dataAir['Month'], inplace=True)
scaler = StandardScaler()
passengers = dataAir['#Passengers'].values.reshape((-1, 1))
scaler.fit(passengers)
dataAir['#Passengers'] = scaler.transform(passengers)
dataAir['before1'] = dataAir['#Passengers'].shift(1)
# dataAir['before1'] = 0
dataAir['beforeLag'] = dataAir['#Passengers'].shift(lag)


## test/train split
timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1958-12-01')]
timeTest = [pd.to_datetime('1959-01-01') , pd.to_datetime('1960-01-01')]

dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
dataTest = dataAir[timeTest[0]: timeTest[1]]


# ## remove NA
# filterNA = dataTrain['beforeLag'] == dataTrain['beforeLag']
# dataTrain = dataTrain[filterNA]

## split features and targets
# xTrain = dataTrain[[ 'beforeLag']].fillna(0).to_numpy()#.values
xTrain = dataTrain[['before1', 'beforeLag']].fillna(0).to_numpy()#.values
yTrain = dataTrain['#Passengers'].values

# xTest = dataTest[['beforeLag']].fillna(0).to_numpy()#.values
xTest = dataTest[['before1', 'beforeLag']].fillna(0).to_numpy()#.values
yTest = dataTest['#Passengers'].values


## Build/fit model 
# model_lgbm = LGBMRegressor(num_leaves=100, max_depth=200, learning_rate=0.1, n_estimators=100)
# model_lgbm = LGBMRegressor(num_leaves=100, max_depth=200, learning_rate=0.1, n_estimators=100)
model_lgbm = XGBRegressor()
# model_lgbm = LinearRegression()
lgbm = model_lgbm.fit(xTrain, yTrain)

## evaluate test the model 
predTest = lgbm.predict(xTest)
predTrain = lgbm.predict(xTrain)
error = mean_squared_error(predTest, yTest)

print(error)

plt.plot(yTest)
plt.plot(predTest)
plt.show()

# plt.plot(yTrain)
# plt.plot(predTrain)
# plt.show()


# plot values 
# plt.plot(predTest)
# plt.plot(yTest)
# plt.plot(xTest[:,1])
# plt.show()

# plt.plot(yTrain)
# plt.plot(xTrain[:,1])
# plt.show()



# ## split train/test data 
# timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
# timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]

# dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
# indexTrain = dataTrain.index
# valTrain = dataTrain['#Passengers'].values
# yTrain = pd.Series(valTrain, indexTrain)

# dataTest = dataAir[timeTest[0]: timeTest[1]]
# indexTest = dataTest.index
# valTest = dataTest['#Passengers'].values
# yTest = pd.Series(valTest, indexTest)

# # build and fit model 
# minValTrain = valTrain.min()-0.1
# modelES = ExponentialSmoothing(yTrain-minValTrain, 
#                                seasonal_periods=lag, 
#                                trend='add',
#                                seasonal='add',
#                                use_boxcox=True,
#                                initialization_method='estimated')
# modelESFitted = modelES.fit()

# # test/evaluate the model  
# lenTest = len(yTest) 
# predTest = modelESFitted.forecast(lenTest) + minValTrain
# error = mean_squared_error(predTest, yTest)
# print(error)

# # plot values 
# plt.plot(predTest)
# plt.plot(yTest)
# plt.show()

# # collect and load data 
# lag = 12
# dataAir = AirPassengersDataset().load().pd_dataframe()

# # pre-process data 
# scaler = StandardScaler()
# passengers = dataAir['#Passengers'].values.reshape((-1,1))
# scaler.fit(passengers)
# dataAir['#Passengers'] = scaler.transform(passengers)
# dataAir['before1'] = dataAir['#Passengers'].shift(1)
# dataAir['beforeLag'] = dataAir['#Passengers'].shift(lag)

# ## split train/test 
# timeTrain = [pd.to_datetime('1950-01-01'), pd.to_datetime('1957-12-01')]
# timeTest = [pd.to_datetime('1958-01-01'), pd.to_datetime('1958-12-01')]

# dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
# X_train = dataTrain[['before1', 'beforeLag']].fillna(0).to_numpy()
# y_train = dataTrain['#Passengers'].values

# dataTest = dataAir[timeTest[0]: timeTest[1]]
# X_test = dataTest[['before1', 'beforeLag']].fillna(0).to_numpy()
# y_test = dataTest['#Passengers'].values

# ## Build/fit model 
# model_lgbm = LGBMRegressor()
# lgbm = model_lgbm.fit(X_train, y_train)

# ## evaluate test the model 
# pred = lgbm.predict(X_test)
# error = mean_squared_error(pred, y_test)

# print(error)


# # parameters 
# windowSize = 3
# lag = 11


# # data collection and loading data 
# data_folder = "../../../data"
# filename = 'AirPassengers.csv'
# dataAir = pd.read_csv(f"{data_folder}/{filename}")


# # data pre-processing/preparation # data already preprocessed here
# dataAir['Month'] = pd.to_datetime(dataAir['Month'])
# dataAir.set_index(dataAir['Month'], inplace=True)
# scaler = StandardScaler()
# passengers = dataAir['#Passengers'].values.reshape((-1,1))
# scaler.fit(passengers)
# dataAir['#Passengers'] = scaler.transform(passengers)

# ## prepare features
# dataAir['MA'] = dataAir['#Passengers'].rolling(windowSize).mean()
# dataAir['MALag'] = dataAir['MA'].shift(lag)

# ## test/train split
# timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
# timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]

# dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
# dataTest = dataAir[timeTest[0]: timeTest[1]]

# ## remove NA
# filterNA = dataTrain['MALag'] == dataTrain['MALag']
# dataTrain = dataTrain[filterNA]

# ## split features and targets
# xTrain = dataTrain[['MALag']].values
# yTrain = dataTrain['#Passengers'].values

# xTest = dataTest[['MALag']].values
# yTest = dataTest['#Passengers'].values


# # build and fit model 
# modelLR = LinearRegression()
# # modelLR.fit(xTrain, yTrain)
# modelLR.fit(xTest, yTest)


# # test/evaluate the model  
# predTest = modelLR.predict(xTest)
# error = mean_squared_error(predTest, yTest)
# print(error)


# error1 = mean_squared_error(xTest*1.1, yTest)
# print(error1)


# ma = dataAir['#Passengers'].rolling(windowSize).mean()

# pred = dataAir['#Passengers'].copy()
# pred.iloc[windowSize+lag:] = ma.iloc[windowSize:-lag].values * 1.1 # try to estimate this parameter
# pred.iloc[:windowSize+lag] = np.nan


# time_test = [pd.to_datetime('1958-01-01'), pd.to_datetime('1959-01-01')]

# y_test = dataAir['#Passengers'][time_test[0]: time_test[1]].values
# pred_test =  pred[time_test[0]: time_test[1]].values
# error2 = mean_squared_error(y_test, pred_test)

# print(error2)

# plt.plot(y_test)
# plt.plot(pred_test)
# plt.show()