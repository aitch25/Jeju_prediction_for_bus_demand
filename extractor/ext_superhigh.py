import sys
import pandas as pd


if __name__=="__main__":
    train = pd.read_csv('./DAT/origin/train.csv')

    train['NewKey'] = train['bus_route_id'] * 10000 + train['station_code']
    train['SuperHigh'] = 0

    #train.loc[train['18~20_ride'] > 50, ['SuperHigh']] = 1
    #train.loc[train['18~20_ride'] > 100, ['SuperHigh']] = 2
    train.loc[train['18~20_ride'] > 20, ['SuperHigh']] = 1
    #train.loc[train['18~20_ride'] > 200, ['SuperHigh']] = 4
    #train.loc[train['18~20_ride'] > 20, ['SuperHigh']] = 1
    print(train[train['SuperHigh'] == 1].shape)


    #train.loc[train['18~20_ride'] > 200, ['SuperHigh']] = 4

    train = train[['id', 'NewKey', 'SuperHigh']]

    train.to_csv('./DAT/TrainDat/10_SuperHigh.csv', index=False)



    test = pd.read_csv('./DAT/origin/test.csv')

    test['NewKey'] = test['bus_route_id'] * 10000 + test['station_code']

    test = pd.merge(test, train[['NewKey', 'SuperHigh']], how='left', left_on='NewKey', right_on='NewKey')

    test = test[['id', 'NewKey', 'SuperHigh']]

    test.to_csv('./DAT/TestDat/10_SuperHigh.csv', index=False)

