import sys

CLASS_PATH = './classes'
sys.path.insert(0, CLASS_PATH)

import csv
import pandas as pd 
import numpy as np 
import pickle 
import sklearn.ensemble as ske 
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from xgboost import XGBClassifier 
import os
import json
#import time as t
from time import sleep

import xgboost as xgb

import Experiment_Test as test
import matplotlib.pyplot as plt

from tqdm import tqdm

def ansTune(mRef_Path, mAns):
    #test = pd.read_csv(mRef_Path)
    test = selectFeature(mRef_Path)

    test['new_18~20_ride'] = mAns

    ansDic = dict()
    ansDic['id'] = list()
    ansDic['18~20_ride'] = list()
    for bns in tqdm(list(set(test['BnS']))):
        r = (test[test['BnS']==bns]['BnS10_mean'].mean() - test[test['BnS']==bns]['new_18~20_ride'].mean())

        for idV, ans in zip(list(test[test['BnS']==bns]['id']), list(test[test['BnS']==bns]['new_18~20_ride']+r)):
            ansDic['id'].append(idV)
            if ans < 0:
                ansDic['18~20_ride'].append(0)
            else:
                ansDic['18~20_ride'].append(ans)

    newAns = pd.DataFrame.from_dict(ansDic)
    newAns = newAns.sort_values(by=['id'])
    #print(newAns[['id', '18~20_ride']])
    
    #newAns[['id', '18~20_ride']].to_csv('jeju_answer_1204_3_Tune.csv', index=False)
    return newAns[['18~20_ride']]
    
def ansTune_GP(mRef_Path, mAns):

    #test = pd.read_csv(mRef_Path)
    test = selectFeature(mRef_Path)
    test['new_18~20_ride'] = mAns

    ansDic = dict()
    ansDic['id'] = list()
    ansDic['18~20_ride'] = list()
    for bns in tqdm(list(set(test['BnS']))):

        #minV = test[test['BnS']==bns]['BnS10_min'].min()
        minV = test[test['BnS']==bns]['BnS10_25%'].min()
        #maxV = test[test['BnS']==bns]['BnS10_max'].max()
        maxV = test[test['BnS']==bns]['BnS10_75%'].max()

        if minV == maxV:
            r = (test[test['BnS']==bns]['BnS10_mean'].mean() - test[test['BnS']==bns]['new_18~20_ride'].mean())
            gp_ans = list(test[test['BnS']==bns]['new_18~20_ride']+r)

        else:
            #print(minV, maxV)
            sc = preprocessing.MinMaxScaler((minV, maxV))
            #print('1', list(test[test['BnS']==bns]['new_18~20_ride'].values))
            origins = np.transpose(test[test['BnS']==bns]['new_18~20_ride'].values).reshape(-1, 1)
            #print('2', origins)
            gp_ans = sc.fit_transform(origins)
            #print('3', list(gp_ans[0]))
            #sleep(1)

        for idV, ans in zip(list(test[test['BnS']==bns]['id']), gp_ans):
            ansDic['id'].append(idV)
            if ans < 0:
                ansDic['18~20_ride'].append(0)
            else:
                ansDic['18~20_ride'].append(ans)

    newAns = pd.DataFrame.from_dict(ansDic)
    newAns = newAns.sort_values(by=['id'])
    #print(newAns[['id', '18~20_ride']])
    
    #newAns[['id', '18~20_ride']].to_csv('jeju_answer_1204_2_GP.csv', index=False)
    return newAns[['18~20_ride']]


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
            feats_forUse = feats_forUse.sort_values(by=['id'])
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

def split_byLab(mData):
    #trainDat = selectFeature('./DAT/TrainDat/')
    trainDat = selectFeature('./DAT/splitted_train/')
    conDat = list(set(trainDat[trainDat['18~20_ride'] > 10]['BnS']))

    data = mData
    for con in tqdm(conDat):
        data = data.drop(data[data['BnS']==con].index)

    print('ret Shape:', data.shape)

    return data
        


#def writeAns(mInPath, mOutName, mAns, mLabel):
def writeAns(mInPath, mOutName, mAns):
    with open(mOutName, 'w') as fw:
        #virList = pd.read_csv(mInPath, sep=',')
        virList = pd.read_csv(mInPath)
        #print(virList['id'].values)

        virL = virList['id'].values

        wr = csv.writer(fw)
        #wr.writerow(['id', '18~20_ride', 'Label'])
        wr.writerow(['id', '18~20_ride'])

        #for vL, a, l in zip(virL, mAns, mLabel):
        for vL, a in zip(virL, mAns):
            wrLst = list()
            wrLst.append(vL)
            wrLst.append(float(a))

            wr.writerow(wrLst)



def jobLoader(mJob_Name):
    clf = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'classifier_Reg/' + mJob_Name + '_classifier.pkl'))
    features = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'classifier_Reg/' + mJob_Name + '_feats.pkl'), 'rb'))
    return clf, features



if __name__=="__main__":

    test_path = sys.argv[1]

    data = selectFeature(test_path)
    data = split_byLab(data)
    data = data.sort_values(['id'])

    y_True = False

    try:
        y = data['18~20_ride'].values
        y_True = True
    except: pass

    resDic = dict()
    resLst = list()

    
    for n in range(0):
        fClf, features = jobLoader('Alg1_' + str(n))
    
        
        
        Xt = data[features].values

        sc = preprocessing.MinMaxScaler()
        Xt = sc.fit_transform(Xt)

        res = fClf.predict(Xt)
        resLst.append(res)
        print("Alg1 Done")

        if y_True:
            rmse = mean_squared_error(y, res)
            print("\nRMSE : ", rmse)
       
    for n in range(3):
        fClf, features = jobLoader('Alg2_' + str(n))

        Xt = data[features].values

        sc = preprocessing.MinMaxScaler()
        Xt = sc.fit_transform(Xt)

        res = fClf.predict(Xt)
        resLst.append(res)

        print("Alg2 Done")

        if y_True:
            rmse = mean_squared_error(y, res)
            print("\nRMSE : ", rmse)
    
    for n in range(0):
        fClf, features = jobLoader('Alg3_' + str(n))

        Xt = data[features].values

        sc = preprocessing.MinMaxScaler()
        Xt = sc.fit_transform(Xt)

        res = fClf.predict(Xt)
        resLst.append(res)
        print("Alg3 Done")

        if y_True:
            rmse = mean_squared_error(y, res)
            print("\nRMSE : ", rmse)

    overallRes = np.transpose(resLst)
    calcRes = list()
    for oRes in overallRes:
        res = np.average(oRes)
        calcRes.append(res)
    calcRes = np.array(calcRes)
    calcRes_GP = ansTune_GP(test_path, calcRes)
    calcRes_Tune = ansTune(test_path, calcRes)

    calcRes = np.where(calcRes<0.1, 0, calcRes)
    calcRes2 = np.where(calcRes>180, calcRes+5, calcRes)

    calcRes_GP_Wt = np.where(calcRes_GP<0.1, 0, calcRes_GP)
    calcRes_Tune_Wt = np.where(calcRes_Tune<0.1, 0, calcRes_Tune)


    if y_True:
        print("\n -> overall RMSE (flt): ", mean_squared_error(y, calcRes))
        print("\n -> overall RMSE (Wt): ", mean_squared_error(y, calcRes2))

        print("\n -> overall RMSE (GP): ", mean_squared_error(y, calcRes_GP))
        print("\n -> overall RMSE (GP_Wt): ", mean_squared_error(y, calcRes_GP_Wt))

        print("\n -> overall RMSE (Tune): ", mean_squared_error(y, calcRes_Tune))
        print("\n -> overall RMSE (Tune_Wt): ", mean_squared_error(y, calcRes_Tune_Wt))

        print("\n -> overall RMSE (int): ", mean_squared_error(y, np.round(calcRes)))

        #plt.subplot(2, 1, 1)
        plt.figure()
        plt.plot(y[calcRes>30], label='y ans')
        plt.plot(calcRes[calcRes>30], label='Overall Res (flt)')
        plt.axis([0, len(y), 0, 300])
        plt.legend(loc='upper right')
        plt.grid(True, which='both')

        #plt.subplot(2, 1, 2)
        #plt.plot(np.array(calcRes), label='Overall Res (flt)')
        #plt.axis([0, len(y), 0, 300])
        #plt.legend(loc='upper right')
        #plt.grid(True, which='both')

        plt.savefig('./figs/overall.png')


    idFile = os.listdir(test_path)
    idFile.sort()
    writeAns(test_path + idFile[0], './jeju_answer_1205LX_1_Reg.csv', np.array(calcRes).clip(min=0))
    writeAns(test_path + idFile[0], './jeju_answer_1205LX_2Tun_Reg.csv', np.array(calcRes_Tune_Wt).clip(min=0))
    writeAns(test_path + idFile[0], './jeju_answer_1205LX_3GP_Reg.csv', np.array(calcRes_GP_Wt).clip(min=0))
    #writeAns(test_path + idFile[0], './jeju_answer_1202_3_Reg.csv', np.array(calcRes))
    print("=======================================================================\n\n")

 

