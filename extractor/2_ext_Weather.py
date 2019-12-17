import sys
import pandas as pd
from time import sleep


def selectFeature(mFeatLst):

    dirs = os.listdir(mFeatLst)
    dirs.sort()
    print("FeatureSet:", dirs)

    key = mFeatLst.split('/')[-2]
    key = key.split('_')[0]

    feats_forUse = pd.DataFrame()
    for d in dirs:
        dirPath = mFeatLst + d

        if feats_forUse.empty:
            print("Init:", d)
            feats_forUse = pd.read_csv(dirPath)
        
            feats_forUse = feats_forUse[['date', 'time', key]]

        else:
            print("Merge:", d)
            forMerge = pd.read_csv(dirPath)
            forMerge = forMerge[key]

            feats_forUse = pd.merge(feats_forUse, forMerge, left_index=True, right_index=True)
            print(feats_forUse.shape)


    return feats_forUse


def df_describe(mPath):
    df = pd.read_csv(mPath)

    key_Df = df[['id', 'date', 'time']]
    main_Df = pd.read_csv(mPath).drop(['id', 'date', 'time'], axis=1)

    out = main_Df.T
    out = out.describe().T
    out = out.drop(['count'], axis=1)

    out = pd.merge(key_Df, out, left_index=True, right_index=True)

    return out


def data_reFormat(mData, mDate):

    mData = mData[mData['date'] == mDate]

    timeRange = list(mData['time'])
    mData = mData.drop(['id', 'date', 'time'], axis=1)
    keys = list(mData.keys())

    head = list()

    for key in keys:
        for tr in timeRange:
            head.append(key + "_" + str(int(tr/100)))

    outDf = pd.DataFrame(columns=head)

    #print(mData['mean'].shape)

    forAppend = list(mData['mean'])
    forAppend.extend(list(mData['std']))
    forAppend.extend(list(mData['min']))
    forAppend.extend(list(mData['25%']))
    forAppend.extend(list(mData['50%']))
    forAppend.extend(list(mData['75%']))
    forAppend.extend(list(mData['max']))

    idx_df = pd.DataFrame({'date':head})
    app_df = pd.DataFrame({mDate:forAppend})

    #print(app_df)
    return (pd.merge(idx_df, app_df, left_index=True, right_index=True))



if __name__=="__main__":

    trainDat = pd.read_csv(sys.argv[1])
    data = (df_describe(sys.argv[2]))

    merged = pd.merge(data_reFormat(data, '2019-10-01'), data_reFormat(data, '2019-10-02'), left_on='date', right_on='date')

    dateLst = list(set(trainDat['date']))
    dateLst.sort()

    for date in dateLst[2:]:
        merged = pd.merge(merged, data_reFormat(data, date), left_on='date', right_on='date')

    merged = merged.set_index('date')
    merged = merged.T

    merged.index.name = 'date'

    #merged.to_csv('./DAT/TrainDat/7_' + sys.argv[2].split('/')[-1], index=True)

    merge_withMain = pd.merge(trainDat[['id', 'date']], merged, left_on='date', right_on='date')
    #print(merge_withMain)
    merge_withMain = merge_withMain.drop(['date'], axis=1)
    merge_withMain.to_csv('./DAT/TestDat/7_' + sys.argv[2].split('/')[-1], index=False)












