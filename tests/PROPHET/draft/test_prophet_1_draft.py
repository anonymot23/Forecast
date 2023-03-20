import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import cmdstanpy
# cmdstanpy.install_cmdstan()
# cmdstanpy.install_cmdstan(compiler=True)# only valid on Windows

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


# # parameters
# lag = 12

# # collect and load data 
# data_folder = "../../data"
# filename = 'AirPassengers.csv'
# dataAir = pd.read_csv(f"{data_folder}/{filename}")


# # prepare/pre-process data 
# dataAir['Month'] = pd.to_datetime(dataAir['Month'])
# dataAir.set_index(dataAir['Month'], inplace=True)
# scaler = StandardScaler()
# passengers = dataAir['#Passengers'].values.reshape((-1, 1))
# scaler.fit(passengers)
# dataAir['#Passengers'] = scaler.transform(passengers)

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


# scaler = StandardScaler()
# scaler.fit(data)
# scaled = scaler.transform(data)
# print(scaled)

# # for inverse transformation
# inversed = scaler.inverse_transform(scaled)
# print(inversed)

if __name__ == "__main__":
    # parameters 
    windowSize = 3
    shift = 1
    lag = 12
    
    
    # data collection and loading data 
    data_folder = "../../../data"
    filename = 'AirPassengers.csv'
    dataAir = pd.read_csv(f"{data_folder}/{filename}")
    
    
    # data pre-processing/preparation # data already preprocessed here
    dataAir['Month'] = pd.to_datetime(dataAir['Month'])
    dataAir.set_index(dataAir['Month'], inplace=True)
    scaler = StandardScaler()
    passengers = dataAir['#Passengers'].values.reshape((-1,1))
    scaler.fit(passengers)
    dataAir['#Passengers'] = scaler.transform(passengers)
    
    ## prepare features
    dataAir['Before1'] = dataAir['#Passengers'].shift(shift)
    dataAir['BeforeLag'] = dataAir['#Passengers'].shift(lag)
    # dataAir['MA'] = dataAir['#Passengers'].rolling(windowSize).mean()
    # dataAir['MALag'] = dataAir['MA'].shift(lag)
    
    ## test/train split
    timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
    timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]
    
    dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
    dataTest = dataAir[timeTest[0]: timeTest[1]]
    
    # reformat data
    yTrain = dataTrain['#Passengers'].values
    df_train = pd.DataFrame.from_dict({"ds": dataTrain.index, "y": yTrain.flatten()})

    yTest = dataTest['#Passengers'].values
    df_test = pd.DataFrame.from_dict({"ds": dataTest.index, "y": yTest.flatten()})

    
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
    
    
    # plt.plot(y_test)
    # plt.plot(predTest)
    # plt.show()
    
    
    # plt.plot(dataAir['#Passengers'])
    # plt.plot(dataAir['MALag']*1.1)
    # plt.show()
    
    
    # timeTrain = [pd.to_datetime('1950-01-01'), pd.to_datetime('1957-12-01')]
    # timeTest = [pd.to_datetime('1958-01-01'), pd.to_datetime('1959-01-01')]
    # lag = 12
    
    # scaler = StandardScaler()
    # passengers = dataAir['#Passengers'].values.reshape((-1,1))
    # scaler.fit(passengers)
    # dataAir['#Passengers'] = scaler.transform(passengers)
    # dataAir['Before1'] = dataAir['#Passengers'].shift(1)
    # dataAir['BeforeLag'] = dataAir['#Passengers'].shift(lag)
    
    
    # dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
    # X_train = dataTrain[['Before1', 'BeforeLag']].fillna(0).values
    # y_train = dataTrain['#Passengers'].values
    # X_train_rescaled = StandardScaler().fit_transform(X_train)
    # y_train_rescaled = StandardScaler().fit_transform(y_train.reshape(-1,1))
    
    # dataTest = dataAir[timeTest[0]: timeTest[1]]
    # X_test = dataTest[['Before1', 'BeforeLag']].fillna(0).values
    # y_test = dataTest['#Passengers'].values
    # X_test_rescaled = StandardScaler().fit_transform(X_test)
    # y_test_rescaled = StandardScaler().fit_transform(y_test.reshape(-1,1))
    
    
    # # prepare inputs bis
    # df_train = pd.DataFrame.from_dict({"ds": dataTrain.index, "y": y_train_rescaled.flatten()})
    # df_test = pd.DataFrame.from_dict({"ds": dataTest.index, "y": y_test_rescaled.flatten()})
    
    
    # train model
    model_proph = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=1)
    # model_proph = Prophet()
    model_proph.fit(df_train)
    # model_proph.fit(df_test)
    y_res_proph = model_proph.predict(df_test[["ds"]])
    pred_test = y_res_proph['yhat']
    
    error2 = mean_squared_error(yTest, pred_test)
    print(error2)
    
    plt.plot(yTest)
    plt.plot(pred_test)
    plt.show()
    
    
    # plt.plot(yTest)
    # plt.plot(pred_test)
    # plt.show()
    