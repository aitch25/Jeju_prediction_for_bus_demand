import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm
from time import sleep

def check_Kinds(mData, mMon):

    kinds_Dict = dict()
    kinds_Dict['date'] = list()
    kinds_Dict['kinds'] = list()


    if mMon == '09':
        maxDay = 31

    elif mMon == '10':
        maxDay = 17


    for plot in range(1, maxDay):
        date = str('2019-%02d-%02d' % (int(mMon), plot))
        newDat = mData[mData['date'] == date]
        kinds_Dict['date'].append(date)
        kinds_Dict['kinds'].append(newDat.shape[0])

    outDict = pd.DataFrame.from_dict(kinds_Dict)

    return outDict


def check_OverAllSum(mData, mOption, mMon):

    oas_Dict = dict()
    oas_Dict['date'] = list()
    oas_Dict['oaSum'] = list()

    if mMon == '09':
        maxDay = 31

    elif mMon == '10':
        maxDay = 17


    for plot in range(1, maxDay):
        date = str('2019-%02d-%02d' % (int(mMon), plot))
        newData = mData[mData['date'] == date]

        if mOption=='ride':
            newData = newData[['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']]

        elif mOption=='toff':
            newData = newData[['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']]
        
        oas_Dict['date'].append(date)
        oas_Dict['oaSum'].append(newData.sum().sum())

    outDict = pd.DataFrame.from_dict(oas_Dict)

    return outDict
         


if __name__=="__main__":
    train = pd.read_csv('./DAT/origin/train.csv')
    test = pd.read_csv('./DAT/origin/test.csv')

    #print(check_Kinds(train))

    #print(train)
    merged_kinds = pd.merge(train[['id', 'date']], check_Kinds(train, '09'), how='left', left_on='date', right_on='date')
    merged_oas = pd.merge(merged_kinds, check_OverAllSum(train, 'ride', '09'), how='left', left_on='date', right_on='date')

    merged = merged_oas.drop(['date'], axis=1)
    merged.to_csv('./DAT/TrainDat/9_kinds_n_oas.csv', index=False)


    merged_kinds = pd.merge(test[['id', 'date']], check_Kinds(test, '10'), how='left', left_on='date', right_on='date')
    merged_oas = pd.merge(merged_kinds, check_OverAllSum(test, 'ride', '10'), how='left', left_on='date', right_on='date')

    merged = merged_oas.drop(['date'], axis=1)
    merged.to_csv('./DAT/TestDat/9_kinds_n_oas.csv', index=False)

