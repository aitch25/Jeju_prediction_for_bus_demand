
import sys
import pandas as pd

if __name__=="__main__":
    train = pd.read_csv('./DAT/origin/train.csv')
    test = pd.read_csv('./DAT/origin/test.csv')

    #newTrain_s = train['id', 'station_code']
    #newTrain_b = train['id', 'bus_route_id']

    pd.get_dummies(train[['id', 'station_code']], columns = ['station_code']).to_csv('./DAT/TrainDat/15_station_oh.csv', index=False)
    pd.get_dummies(train[['id', 'bus_route_id']], columns = ['bus_route_id']).to_csv('./DAT/TrainDat/16_route_oh.csv', index=False)


    #newTest_s = test['id', 'station_code']
    #newTest_b = test['id', 'bus_route_id']

    pd.get_dummies(test[['id', 'station_code']], columns=['station_code']).to_csv('./DAT/TestDat/15_station_oh.csv', index=False)
    pd.get_dummies(test[['id', 'bus_route_id']], columns=['bus_route_id']).to_csv('./DAT/TestDat/16_route_oh.csv', index=False)


