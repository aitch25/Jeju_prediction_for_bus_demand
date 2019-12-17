import os
import sys
from tqdm import tqdm
import pandas as pd
from time import sleep


def userTypes(mMainDat, mBusDat, mMonth):
    #stations = list(set(mBusDat['geton_station_code']))

    mBusDat['BnS'] = (mBusDat['bus_route_id']*10000 + mBusDat['geton_station_code'])
    BnS = list(set(mBusDat['bus_route_id']*10000 + mBusDat['geton_station_code']))
    dates = list(set(mBusDat['geton_date']))
    dates.sort()

    out_catDf = pd.DataFrame()
    out_cntDf = pd.DataFrame()

    for date in dates:
        ft_1_df = mBusDat[mBusDat['geton_date'] == date]

        month = date.split('-')[1]
        if not (month == mMonth): continue

        #for station in tqdm(stations):
        for bns in tqdm(BnS):
            #ft_2_df = ft_1_df[ft_1_df['geton_station_code'] == station]
            ft_2_df = ft_1_df[ft_1_df['BnS'] == bns]
        
            #catDf = pd.DataFrame({'date' : [date], 'station_code' : [station]})
            catDf = pd.DataFrame({'date' : [date], 'BnS' : [bns]})
            #cntDf = pd.DataFrame({'date' : [date], 'station_code' : [station]})
            cntDf = pd.DataFrame({'date' : [date], 'BnS' : [bns]})

            for hour in range(6, 12):
                h_s = str("%02d:00:00" % int(hour))
                h_d = str("%02d:00:00" % int(hour+1))
                ft_3_df = ft_2_df[(ft_2_df['geton_time'] >= h_s) & (ft_2_df['geton_time'] <= h_d)]

                oh_userCnt = list(ft_3_df['user_count'])
                oh_userCat = list(ft_3_df['user_category'])

                cats = ['1', '2', '4', '6', '27', '28', '29', '30']
                cnts = list(range(1, 11))

                for cat in cats:
                    catDf[str('uCat_%02d_%02d' % (hour, int(cat)))] = [oh_userCat.count(int(cat))]

                for cnt in cnts:
                    cntDf[str('uCnt_%02d_%02d' % (hour, cnt))] = [oh_userCnt.count(int(cnt))]

            out_catDf = out_catDf.append(catDf)
            out_cntDf = out_cntDf.append(cntDf)

    #oMainCat = mMainDat[['id', 'date', 'station_code']]
    oMainCat = mMainDat[['id', 'date']]
    oMainCat['BnS'] = oMainCat['bus_route_id']*10000 + oMainCat['station_code']

    #oMainCnt = mMainDat[['id', 'date', 'station_code']]
    oMainCnt = mMainDat[['id', 'date']]
    oMainCnt['BnS'] = oMainCnt['bus_route_id']*10000 + oMainCnt['station_code']

    #oMainCat = pd.merge(oMainCat, out_catDf, how='left', left_on=['date', 'station_code'], right_on=['date', 'station_code'])
    oMainCat = pd.merge(oMainCat, out_catDf, how='left', left_on=['date', 'BnS'], right_on=['date', 'BnS'])
    #oMainCnt = pd.merge(oMainCnt, out_cntDf, how='left', left_on=['date', 'station_code'], right_on=['date', 'station_code'])
    oMainCnt = pd.merge(oMainCnt, out_cntDf, how='left', left_on=['date', 'BnS'], right_on=['date', 'BnS'])

    #oMainCat = oMainCat.drop(['date', 'station_code'], axis=1)
    oMainCat = oMainCat.drop(['date', 'BnS'], axis=1)
    #oMainCnt = oMainCnt.drop(['date', 'station_code'], axis=1)
    oMainCnt = oMainCnt.drop(['date', 'BnS'], axis=1)

    if mMonth=='09':
        oMainCat.to_csv('./DAT/TrainDat/10_userCat.csv', index=False) 
        oMainCnt.to_csv('./DAT/TrainDat/11_userCnt.csv', index=False)
    else:
        oMainCat.to_csv('./DAT/TestDat/10_userCat.csv', index=False) 
        oMainCnt.to_csv('./DAT/TestDat/11_userCnt.csv', index=False)


if __name__=="__main__":
    trainDat = pd.read_csv('./DAT/origin/train.csv')
    testDat = pd.read_csv('./DAT/origin/test.csv')
    busDat = pd.read_csv('./DAT/origin/bus_bts.csv')

    userTypes(trainDat, busDat, '09')
    userTypes(testDat, busDat, '10')



