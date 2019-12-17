import sys
from tqdm import tqdm
import pandas as pd
from time import sleep

if __name__=="__main__":
    datPath = sys.argv[1]
    option = sys.argv[2]

    mainDat = pd.read_csv(datPath)

    mainDat['68a'] = mainDat['6~7_ride'] + mainDat['7~8_ride'] # 6 ~ 8시 승차인원
    mainDat['79a'] = mainDat['7~8_ride'] + mainDat['8~9_ride']
    mainDat['810a'] = mainDat['8~9_ride'] + mainDat['9~10_ride']
    mainDat['911a'] = mainDat['9~10_ride'] + mainDat['10~11_ride']
    mainDat['1012a'] = mainDat['10~11_ride'] + mainDat['11~12_ride']

    mainDat['68b'] = mainDat['6~7_takeoff'] + mainDat['7~8_takeoff'] # 6 ~ 8시 하차인원
    mainDat['79b'] = mainDat['7~8_takeoff'] + mainDat['8~9_takeoff'] # 6 ~ 8시 하차인원
    mainDat['810b'] = mainDat['8~9_takeoff'] + mainDat['9~10_takeoff']
    mainDat['911b'] = mainDat['9~10_takeoff'] + mainDat['10~11_takeoff']
    mainDat['1012b'] = mainDat['10~11_takeoff'] + mainDat['11~12_takeoff']


    if option=='ride':
        #mainDat = mainDat[['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']]   
        mainDat = mainDat[['68a', '79a', '810a', '911a', '1012a']]   

    elif option=='toff':
        #mainDat = mainDat[['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']]   
        mainDat = mainDat[['68b', '79b', '810b', '911b', '1012b']]   

    
    idxLst = list(range(0, mainDat.shape[0], 1000))
    idxLst.append(mainDat.shape[0])
    
    out_sum_df = pd.DataFrame(columns=['sum'])
    #out_kinds_df = pd.DataFrame(columns=['kinds'])
    out_desc_df = pd.DataFrame(columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

    for n in tqdm(range(0, len(idxLst)-1)):
        mainDat_T = mainDat[idxLst[n]:idxLst[n+1]].T
        new_df = mainDat_T.sum().T
        new_df = pd.DataFrame({'sum': new_df})

        out_sum_df = out_sum_df.append(new_df)

        new_df = mainDat_T.describe().T
        out_desc_df = out_desc_df.append(new_df)




    out_sum_df.index.name = 'id'
    out_desc_df.index.name = 'id'


    if option=='ride':
        if 'train' in datPath:
            print('wr train ' + option)
            out_sum_df.to_csv('./DAT/TrainDat/3_' + option + '_sum.csv', header=['r_sum'])
            out_desc_df.to_csv('./DAT/TrainDat/5_' + option + '_describe.csv', header=['r_count', 'r_mean', 'r_std', 'r_min', 'r_25%', 'r_50%', 'r_75%', 'r_max'])

        elif 'test' in datPath:
            print('wr test ' + option)
            out_sum_df.to_csv('./DAT/TestDat/3_' + option + '_sum.csv', header=['r_sum'])
            out_desc_df.to_csv('./DAT/TestDat/5_' + option + '_describe.csv', header=['r_count', 'r_mean', 'r_std', 'r_min', 'r_25%', 'r_50%', 'r_75%', 'r_max'])



    elif option=='toff':
        if 'train' in datPath:
            print('wr train ' + option)
            out_sum_df.to_csv('./DAT/TrainDat/4_' + option + '_sum.csv', header=['t_sum'])
            out_desc_df.to_csv('./DAT/TrainDat/6_' + option + '_describe.csv', header=['t_count', 't_mean', 't_std', 't_min', 't_25%', 't_50%', 't_75%', 't_max'])

        elif 'test' in datPath:
            print('wr test ' + option)
            out_sum_df.to_csv('./DAT/TestDat/4_' + option + '_sum.csv', header=['t_sum'])
            out_desc_df.to_csv('./DAT/TestDat/6_' + option + '_describe.csv', header=['t_count', 't_mean', 't_std', 't_min', 't_25%', 't_50%', 't_75%', 't_max'])






