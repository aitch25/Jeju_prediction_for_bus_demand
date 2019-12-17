import os

import pandas as pd
import warnings

import geopy.distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor



if __name__=="__main__":
    warnings.filterwarnings('ignore')
    
    train = pd.read_csv("./DAT/origin/train.csv")
    test = pd.read_csv("./DAT/origin/test.csv")
    
    train['date'] = pd.to_datetime(train['date'])
    train['weekday'] = train['date'].dt.weekday
    train = pd.get_dummies(train,columns=['weekday'])
    #train = pd.get_dummies(train,columns=['bus_route_id'])

    test['date'] = pd.to_datetime(test['date'])
    test['weekday'] = test['date'].dt.weekday
    test = pd.get_dummies(test,columns=['weekday'])
    #test = pd.get_dummies(test,columns=['bus_route_id'])

    
    train['in_out'] = train['in_out'].map({'시내':0,'시외':1})
    test['in_out'] = test['in_out'].map({'시내':0,'시외':1})
    
    
    coords_jejusi = (33.500770, 126.522761) #제주시의 위도 경도
    coords_seoquipo = (33.259429, 126.558217) #서귀포시의 위도 경도
    
    
    train['dis_jejusi'] = [geopy.distance.vincenty((train['latitude'].iloc[i],train['longitude'].iloc[i]), coords_jejusi).km for i in range(len(train))]
    train['dis_seoquipo'] = [geopy.distance.vincenty((train['latitude'].iloc[i],train['longitude'].iloc[i]), coords_seoquipo).km for i in range(len(train))]
    
    test['dis_jejusi'] = [geopy.distance.vincenty((test['latitude'].iloc[i],test['longitude'].iloc[i]), coords_jejusi).km for i in range(len(test))]
    test['dis_seoquipo'] = [geopy.distance.vincenty((test['latitude'].iloc[i],test['longitude'].iloc[i]), coords_seoquipo).km for i in range(len(test))]

    
    target=['id', '18~20_ride']
    trainDat = train[target]
    trainDat.to_csv('./DAT/TrainDat/1_Train_Label.csv', index=False)

    drop_var = ['date', 'bus_route_id', 'station_code', 'station_name', '18~20_ride'] #+ ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']
    trainDat = train.drop(drop_var, axis=1)
    trainDat.to_csv('./DAT/TrainDat/2_Train_Baseline.csv', index=False)


    target=['id']
    testDat = test[target]
    testDat.to_csv('./DAT/TestDat/1_Test_Label.csv', index=False)

    drop_var = ['date', 'bus_route_id', 'station_code', 'station_name'] #+ ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']
    testDat = test.drop(drop_var, axis=1)
    testDat.to_csv('./DAT/TestDat/2_Test_Baseline.csv', index=False)



