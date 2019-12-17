import os
import sys
import pandas as pd 
from random import randrange


def selectFeature(mFeatLst):

    dirs = os.listdir(mFeatLst)
    dirs.sort()
    print("FeatureSet:", dirs)

    feats_forUse = pd.DataFrame()
    for d in dirs:
        dirPath = mFeatLst + d

        if feats_forUse.empty:
            print("Init:", d)
            feats_forUse = pd.read_csv(dirPath)
            feats_forUse.sort_values(by=['id'])
            print(feats_forUse.shape)
        else:
            print("Merge:", d)
            forMerge = pd.read_csv(dirPath)
            forMerge = forMerge.drop(['id'], axis=1)

            feats_forUse = pd.merge(feats_forUse, forMerge, how='left', left_index=True, right_index=True)
            print(feats_forUse.shape)
            #print(feats_forUse.keys())


    #feats_forUse['Rt_count'] = feats_forUse['Rt_count'].fillna(1)
    #feats_forUse['Rt_std'] = feats_forUse['Rt_std'].fillna(0)
    #feats_forUse['Rt_mean'] = feats_forUse['Rt_mean'].fillna(10)
    #feats_forUse['Rt_min'] = feats_forUse['Rt_min'].fillna(10)
    #feats_forUse['Rt_25%'] = feats_forUse['Rt_25%'].fillna(10)
    #feats_forUse['Rt_50%'] = feats_forUse['Rt_50%'].fillna(10)
    #feats_forUse['Rt_75%'] = feats_forUse['Rt_75%'].fillna(10)
    #feats_forUse['Rt_max'] = feats_forUse['Rt_max'].fillna(10)

    #feats_forUse['St_count'] = feats_forUse['St_count'].fillna(1)
    #feats_forUse['St_std'] = feats_forUse['St_std'].fillna(0)
    #feats_forUse['St_mean'] = feats_forUse['St_mean'].fillna(10)
    #feats_forUse['St_min'] = feats_forUse['St_min'].fillna(10)
    #feats_forUse['St_25%'] = feats_forUse['St_25%'].fillna(10)
    #feats_forUse['St_50%'] = feats_forUse['St_50%'].fillna(10)
    #feats_forUse['St_75%'] = feats_forUse['St_75%'].fillna(10)
    #feats_forUse['St_max'] = feats_forUse['St_max'].fillna(10)

    #feats_forUse['Rt10_count'] = feats_forUse['Rt10_count'].fillna(1)
    #feats_forUse['Rt10_std'] = feats_forUse['Rt10_std'].fillna(0)
    #feats_forUse['Rt10_mean'] = feats_forUse['Rt10_mean'].fillna(1)
    #feats_forUse['Rt10_min'] = feats_forUse['Rt10_min'].fillna(1)
    #feats_forUse['Rt10_25%'] = feats_forUse['Rt10_25%'].fillna(1)
    #feats_forUse['Rt10_50%'] = feats_forUse['Rt10_50%'].fillna(1)
    #feats_forUse['Rt10_75%'] = feats_forUse['Rt10_75%'].fillna(1)
    #feats_forUse['Rt10_max'] = feats_forUse['Rt10_max'].fillna(1)

    #feats_forUse['St10_count'] = feats_forUse['St10_count'].fillna(1)
    #feats_forUse['St10_std'] = feats_forUse['St10_std'].fillna(0)
    #feats_forUse['St10_mean'] = feats_forUse['St10_mean'].fillna(1)
    #feats_forUse['St10_min'] = feats_forUse['St10_min'].fillna(1)
    #feats_forUse['St10_25%'] = feats_forUse['St10_25%'].fillna(1)
    #feats_forUse['St10_50%'] = feats_forUse['St10_50%'].fillna(1)
    #feats_forUse['St10_75%'] = feats_forUse['St10_75%'].fillna(1)
    #feats_forUse['St10_max'] = feats_forUse['St10_max'].fillna(1)

    feats_forUse = feats_forUse.fillna(0)

    feats_forUse.to_csv('./testfile.csv', index=False)

    return feats_forUse




def randomSample(mData, mSplit=0.2):
    train_Idx = list()
    test_Idx = list()

    testSize = int(round(mData.shape[0] * mSplit))

    while (len(test_Idx) <= testSize):
        randVal = randrange(0, mData.shape[0])
    
        if not randVal in test_Idx:
            test_Idx.append(randVal)

    train_Idx = list(range(mData.shape[0]))
    train_Idx = list(set(train_Idx) - set(test_Idx))


    trainDat = (mData.iloc[lambda x: train_Idx])
    trainDat = trainDat.sort_values(by=['id'])
    trainDat.to_csv('./DAT/splitted_train/trainDat.csv', index=False)

    testDat = (mData.iloc[lambda x: test_Idx])
    testDat = testDat.sort_values(by=['id'])
    testDat.to_csv('./DAT/splitted_test/testDat.csv', index=False)
    
    
if __name__=="__main__":

    try:
        train_path = sys.argv[1]
		

    except:
        print("\n\n\nex 1) python3 run.py <csv file path> ")
        exit()
    

    data = selectFeature(train_path)
    print('Select Done!')
    randomSample(data, 0.2)



    
    print("=======================================================================\n\n")

 

