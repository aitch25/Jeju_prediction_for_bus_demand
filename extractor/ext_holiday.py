import pandas as pd

if __name__=="__main__":

    train = pd.read_csv('./DAT/origin/train.csv')

    dates = pd.date_range('2019-09-01', '2019-09-30', freq='D').to_series()
    dates = dates.dt.weekday

    dates = dates.replace(1, 0)
    dates = dates.replace(2, 0)
    dates = dates.replace(3, 0)
    dates = dates.replace(4, 0)
    dates = dates.replace(5, 1)
    dates = dates.replace(6, 1)

    dates['2019-09-12'] = 1
    dates['2019-09-13'] = 1

    dates.name = 'holiday'

    dates.index = dates.index.strftime('%Y-%m-%d')

    outTrain = pd.merge(train[['id', 'date']], dates, how='left', left_on='date', right_index=True)
    outTrain = outTrain.drop(['date'], axis=1)
    outTrain.to_csv('./DAT/TrainDat/13_holiday.csv', index=False)

##########################################################################################

    test = pd.read_csv('./DAT/origin/test.csv')

    dates = pd.date_range('2019-10-01', '2019-10-20', freq='D').to_series()
    dates = dates.dt.weekday

    dates = dates.replace(1, 0)
    dates = dates.replace(2, 0)
    dates = dates.replace(3, 0)
    dates = dates.replace(4, 0)
    dates = dates.replace(5, 1)
    dates = dates.replace(6, 1)

    dates['2019-10-03'] = 1
    dates['2019-10-09'] = 1

    dates.name = 'holiday'

    dates.index = dates.index.strftime('%Y-%m-%d')

    outTest = pd.merge(test[['id', 'date']], dates, how='left', left_on='date', right_index=True)
    outTest = outTest.drop(['date'], axis=1)
    outTest.to_csv('./DAT/TestDat/13_holiday.csv', index=False)


