import pandas as pd
import numpy as np

if __name__=="__main__":

    train = pd.read_csv('./DAT/origin/train.csv')
    test = pd.read_csv('./DAT/origin/test.csv')
    weather = pd.read_csv('./DAT/origin/weather.csv')

    outTrain = pd.merge(train[['id', 'date']], weather, how='left', left_on='date', right_on='date')
    outTest = pd.merge(test[['id', 'date']], weather, how='left', left_on='date', right_on='date')

    #print(outTrain)
    outTrain = outTrain.drop(['date'], axis=1)
    outTest = outTest.drop(['date'], axis=1)

    outTrain.to_csv('./DAT/TrainDat/12_weather.csv', index=False)
    outTest.to_csv('./DAT/TestDat/12_weather.csv', index=False)


