import numpy as np
import sys
from time import sleep
import pandas as pd
from tqdm import tqdm


if __name__=="__main__":
    
    data = pd.read_csv('DAT/origin/train.csv')
    data_test = pd.read_csv('DAT/origin/test.csv')

    popLst = list([-1]*data.shape[0])
    for n in range(280, 0, -10):
        idx = list(data[(data['18~20_ride'] < n) & (data['18~20_ride'] >= (n-10))]['id'])
        print(n, len(idx))
        for i in idx:
            popLst[i] = n           

    popLst_10 = list([-1]*data.shape[0])
    for n in range(280, 0, -1):
        idx_10 = list(data[(data['18~20_ride'] < n) & (data['18~20_ride'] >= (n-1))]['id'])
        print(n, len(idx_10))
        for i in idx_10:
            popLst_10[i] = (n-1)




    data['popularity'] = popLst
    newDf = pd.DataFrame()
    newDf.index.name = 'station_code'

    for code in list(set(data['station_code'])):
        line = pd.DataFrame({code: data[data['station_code']==code]['popularity'].describe()})
        newDf = newDf.append(line.T)
    newDf = newDf.fillna(0)

    data['pop_10'] = popLst_10
    newDf_10 = pd.DataFrame()
    newDf_10.index.name = 'station_code'

    for code in list(set(data['station_code'])):
        line = pd.DataFrame({code: data[data['station_code']==code]['pop_10']})
        line = line[line[code] > -1].describe()
        newDf_10 = newDf_10.append(line.T)
    newDf_10 = newDf_10.fillna(0)


    #head = ['id', 'station_code', 'St_count', 'St_mean', 'St_std', 'St_min', 'St_25%', 'St_50%', 'St_75%', 'St_max'] + ['St10_count', 'St10_mean', 'St10_std', 'St10_min', 'St10_25%', 'St10_50%', 'St10_75%', 'St10_max']
    #head = ['id', 'St_count', 'St_mean', 'St_std', 'St_min', 'St_25%', 'St_50%', 'St_75%', 'St_max'] + ['St10_count', 'St10_mean', 'St10_std', 'St10_min', 'St10_25%', 'St10_50%', 'St10_75%', 'St10_max']
    head = ['id', 'St10_count', 'St10_mean', 'St10_std', 'St10_min', 'St10_25%', 'St10_50%', 'St10_75%', 'St10_max']

    dataOut = pd.merge(data[['id', 'station_code']], newDf_10, how='left', left_on='station_code', right_index=True)
    #print(dataOut.shape)
    #dataOut = pd.merge(dataOut, newDf_10, how='left', left_on='station_code', right_index=True)
    print(dataOut.shape)
    dataOut = dataOut.drop(['station_code'], axis=1)
    dataOut.to_csv('./DAT/TrainDat/07_St_popularity.csv', header=head, index=False)

    testOut = pd.merge(data_test[['id', 'station_code']], newDf_10, how='left', left_on='station_code', right_index=True)
    #print(testOut.shape)
    #testOut = pd.merge(testOut, newDf_10, how='left', left_on='station_code', right_index=True)
    print(testOut.shape)
    testOut = testOut.drop(['station_code'], axis=1)
    testOut.to_csv('./DAT/TestDat/07_St_popularity.csv', header=head, index=False)






    newDf = pd.DataFrame()
    newDf.index.name = 'bus_route_id'
    for code in list(set(data['bus_route_id'])):
        line = pd.DataFrame({code: data[data['bus_route_id']==code]['popularity'].describe()})
        newDf = newDf.append(line.T)
    newDf = newDf.fillna(0)

    newDf_10 = pd.DataFrame()
    newDf_10.index.name = 'bus_route_id'
    for code in list(set(data['bus_route_id'])):
        line = pd.DataFrame({code: data[data['bus_route_id']==code]['pop_10']})
        line = line[line > -1].describe()
        newDf_10 = newDf_10.append(line.T)
    newDf_10 = newDf_10.fillna(0)
    
    #head = ['id', 'bus_route_id', 'Rt_count', 'Rt_mean', 'Rt_std', 'Rt_min', 'Rt_25%', 'Rt_50%', 'Rt_75%', 'Rt_max'] + ['Rt10_count', 'Rt10_mean', 'Rt10_std', 'Rt10_min', 'Rt10_25%', 'Rt10_50%', 'Rt10_75%', 'Rt10_max']
    #head = ['id', 'Rt_count', 'Rt_mean', 'Rt_std', 'Rt_min', 'Rt_25%', 'Rt_50%', 'Rt_75%', 'Rt_max'] + ['Rt10_count', 'Rt10_mean', 'Rt10_std', 'Rt10_min', 'Rt10_25%', 'Rt10_50%', 'Rt10_75%', 'Rt10_max']
    head = ['id', 'Rt10_count', 'Rt10_mean', 'Rt10_std', 'Rt10_min', 'Rt10_25%', 'Rt10_50%', 'Rt10_75%', 'Rt10_max']

    dataOut = pd.merge(data[['id', 'bus_route_id']], newDf_10, how='left', left_on='bus_route_id', right_index=True)
    #print(dataOut.shape)
    #dataOut = pd.merge(dataOut, newDf_10, how='left', left_on='bus_route_id', right_index=True)
    print(dataOut.shape)
    dataOut = dataOut.drop(['bus_route_id'], axis=1)
    dataOut.to_csv('./DAT/TrainDat/08_Rt_popularity.csv', header=head, index=False)

    testOut = pd.merge(data_test[['id', 'bus_route_id']], newDf_10, how='left', left_on='bus_route_id', right_index=True)
    #print(testOut.shape)
    #testOut = pd.merge(testOut, newDf_10, how='left', left_on='bus_route_id', right_index=True)
    print(testOut.shape)
    testOut = testOut.drop(['bus_route_id'], axis=1)
    testOut.to_csv('./DAT/TestDat/08_Rt_popularity.csv', header=head, index=False)
    
    data['BnS'] = data['bus_route_id']*10000 + data['station_code']
    data_test['BnS'] = data_test['bus_route_id']*10000 + data_test['station_code']

    newDf = pd.DataFrame()
    newDf.index.name = 'BnS'
    for bns in tqdm(list(set(data['BnS']))):
        line = pd.DataFrame({bns: data[data['BnS']==bns]['popularity'].describe()})
        newDf = newDf.append(line.T)
    newDf = newDf.fillna(0)



    newDf_10 = pd.DataFrame()
    newDf_10.index.name = 'BnS'
    for bns in tqdm(list(set(data['BnS']))):
        line = pd.DataFrame({bns: data[data['BnS']==bns]['pop_10']})
        line = line[line > -1].describe()
        newDf_10 = newDf_10.append(line.T)
    newDf_10 = newDf_10.fillna(0)
    
    #head = ['id', 'BnS', 'BnS_count', 'BnS_mean', 'BS_std', 'BnS_min', 'BnS_25%', 'BnS_50%', 'BnS_75%', 'BnS_max'] + ['BnS10_count', 'BnS10_mean', 'BnS10_std', 'BnS10_min', 'BnS10_25%', 'BnS10_50%', 'BnS10_75%', 'BnS10_max']
    #head = ['id', 'BnS_count', 'BnS_mean', 'BS_std', 'BnS_min', 'BnS_25%', 'BnS_50%', 'BnS_75%', 'BnS_max'] + ['BnS10_count', 'BnS10_mean', 'BnS10_std', 'BnS10_min', 'BnS10_25%', 'BnS10_50%', 'BnS10_75%', 'BnS10_max']
    head = ['id', 'BnS', 'BnS10_count', 'BnS10_mean', 'BnS10_std', 'BnS10_min', 'BnS10_25%', 'BnS10_50%', 'BnS10_75%', 'BnS10_max']

    dataOut = pd.merge(data[['id', 'BnS']], newDf_10, how='left', left_on=['BnS'], right_index=True)
    #print(dataOut.shape)
    #dataOut = pd.merge(dataOut, newDf_10, how='left', left_on=['BnS'], right_index=True)
    print(dataOut.shape)
    #dataOut = dataOut.drop(['BnS'], axis=1)
    dataOut.to_csv('./DAT/TrainDat/15_BnS_popularity.csv', header=head, index=False)

    testOut = pd.merge(data_test[['id', 'BnS']], newDf_10, how='left', left_on=['BnS'], right_index=True)
    #print(testOut.shape)
    #testOut = pd.merge(testOut, newDf_10, how='left', left_on=['BnS'], right_index=True)
    print(testOut.shape)
    #testOut = testOut.drop(['BnS'], axis=1)
    testOut.to_csv('./DAT/TestDat/15_BnS_popularity.csv', header=head, index=False)

