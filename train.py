import sys
import copy
import math 
import pandas as pd 
import numpy as np 
import pickle 
import sklearn.ensemble as ske 
from sklearn import tree, linear_model, preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import geopy.distance

import lightgbm as lgb

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

from xgboost import XGBRegressor
import os
import csv
import json
import random as r
import time as t


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping

from datetime import datetime


CLASS_PATH = './classes'
sys.path.insert(0, CLASS_PATH)

class Experiment:
    cAllCases = 10000000000
    cAccThd = 95.0
    	
    def __init__(self, mAccThd):
        self.cAccThd = float(mAccThd)

    def jobStorage(self, mAlg, mJob_Name, mFeatures=None):
        print('Saving algorithm and feature list in classifier directory...')
        joblib.dump(mAlg, './classifier_Reg/' + mJob_Name + '_classifier.pkl')
        pickle.dump(mFeatures, open('./classifier_Reg/' + mJob_Name + '_feats.pkl', 'wb'))
        print('Saved')


    def selectFeature(self, mFeatLst):
    
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

        feats_forUse = feats_forUse.fillna(0)

        print('Final Shape:', feats_forUse.shape)
        
        return feats_forUse


    def sklearn_gSearch(self, mData, mAlgo, mLabType):

        data = mData

        data = data.dropna()

        y = data[mLabType].values

        data = data.drop(['id', '18~20_ride'], axis=1)

        X = data.values
        sc = preprocessing.MinMaxScaler()
        X = sc.fit_transform(X)

        algorithms = {
            "XGB": XGBRegressor(booster="gbtree", verbosity=1, nthread=15, predictor='gpu_predictor'),
            "LGB": lgb.LGBMRegressor(n_jobs=15, random_state=100)
        }


        if mAlgo == 'XGB':
            param_grid = {
                'max_depth': [2, 4, 8, 16, 32, 48, 64],
                #'booster': ['dart', 'gbtree', 'gblinear'],
            }

        elif mAlgo == 'LGB':
            param_grid = {
                #'max_depth': [-1, 16, 48, 64],
                'num_leaves': [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120], #2452,
                'min_child_samples': list(range(0, 55, 5)),
            }


        grid_search = GridSearchCV(estimator = algorithms[mAlgo], param_grid = param_grid) 
        grid_search.fit(X, y)


        print(grid_search.best_params_)


    def sklearn_main(self, mData, mAlgo, mN_est, mLabType):
        data = mData

        data = data.dropna()

        y = data[mLabType].values

        del_count = []#'RepCnt']

        data = data.drop(['id', '18~20_ride'] + del_count, axis=1)

        X = data.values
        sc = preprocessing.MinMaxScaler()
        X = sc.fit_transform(X)
        
        algorithms = {
            "XGB": XGBRegressor(booster="gbtree", n_estimators=mN_est, verbosity=1, nthread=15, max_depth=4, predictor='gpu_predictor'),
            "LGB": lgb.LGBMRegressor(n_jobs=15)
        }

        
        for n in range(self.cAllCases):
            oFeatures = list(data.columns)
            print('feats: ', len(oFeatures))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=False)#r.randrange(50))
            print(y_train, y_test)
            
            if mAlgo == "LGB":
                X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=0, shuffle=False)#r.randrange(50))
                print(mAlgo, y_train.max())

                oClf = algorithms[mAlgo]
                opt_params = {'n_estimators':mN_est,
                                'boosting_type': 'gbdt',
                                'objective': 'regression',
                                'num_leaves': 24, #2452,
                                'min_child_samples': 10, #212,
                                #'device_type':'gpu',
                                'max_depth':-1,
                            }


                oClf.set_params(**opt_params)
                oClf.fit(X_train, y_train, eval_set=(X_eval, y_eval), early_stopping_rounds=500, verbose=100)

                res = oClf.predict(X_test)
                rmse = mean_squared_error(y_test, res)
                rmse_int = mean_squared_error(y_test, np.round(res))

                res_zero = np.where(res<0.1, 0, res)
                
                res_zero2 = np.where(res_zero>180, res_zero+5, res_zero)
                print(len(np.where(res_zero2>180)), res_zero.mean(), res_zero2.mean())

                rmse_zero = mean_squared_error(y_test, res_zero)
                rmse_zero2 = mean_squared_error(y_test, res_zero2)

                plt.figure()
                plt.plot(y_test[res_zero > 30], label='y test', marker='.')
                plt.plot(res_zero[res_zero > 30], label='float res', marker='.')
                plt.legend(loc='upper left')
                now = str(datetime.now()).replace(' ', '')
                now = now.replace('.', '')
                plt.savefig('./figs/' + mAlgo + '_' + now + '.png')

                prt = ("%d) %s RMSE (flt): %f // MAX: %f %%\n" % (n+1, mAlgo, rmse, res.max()))
                print(prt)
                prt = ("%d) %s RMSE (int): %f // MAX: %f %%\n" % (n+1, mAlgo, rmse_int, np.round(res.max())))
                print(prt)
                prt = ("%d) %s RMSE (zero): %f // MAX: %f %%\n" % (n+1, mAlgo, rmse_zero, res_zero.max()))
                print(prt)
                prt = ("%d) %s RMSE (zero2): %f // MAX: %f %%\n" % (n+1, mAlgo, rmse_zero2, res_zero2.max()))
                print(prt)

                print("=======================================================================\n\n")

            if mAlgo == "XGB":
                X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=r.randrange(50))
                print(mAlgo, y_train.max())

                oClf = algorithms[mAlgo]

                oClf.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], early_stopping_rounds=300, verbose=False)

                res = oClf.predict(X_test)
                rmse = mean_squared_error(y_test, res)
                rmse_int = mean_squared_error(y_test, np.round(res))
                res_zero = np.where(res<0.1, 0, res)
                rmse_zero = mean_squared_error(y_test, res_zero)

                plt.figure()
                plt.plot(y_test[res_zero > 1], label='y test', marker='.')
                plt.plot(res_zero[res_zero > 1], label='float res', marker='.')
                plt.legend(loc='upper left')

                now = str(datetime.now()).replace(' ', '')
                now = now.replace('.', '')
                plt.savefig('./figs/' + mAlgo + '_' + now + '.png')


                prt = ("%d) %s RMSE (flt): %f // MAX: %f %%\n" % (n+1, mAlgo, rmse, res.max()))
                print(prt)
                prt = ("%d) %s RMSE (int): %f // MAX: %f %%\n" % (n+1, mAlgo, rmse_int, np.round(res.max())))
                print(prt)
                prt = ("%d) %s RMSE (zero): %f // MAX: %f %%\n" % (n+1, mAlgo, rmse_zero, res_zero.max()))
                print(prt)

                print("=======================================================================\n\n")


            with open('./results_Mul.csv', 'a') as outs:
                outs.write(prt)

            print("=======================================================================\n\n")

            if rmse <= self.cAccThd: 
                return rmse, oClf, oFeatures, res_zero


if __name__=="__main__":
    try:
        csv_path = sys.argv[1]
        
    except:
        print("\n\n\nex 1) python3 run.py <csv file path> ")
        exit()
    
    exp = Experiment(30.0)

    data = exp.selectFeature(csv_path)
   

    accWeight = dict()
    resLst = list()

    weight_Df = pd.DataFrame()

    
    for n in range(0):
        exp.sklearn_gSearch(data, "LGB", '18~20_ride')

    for n in range(0):
        exp.sklearn_gSearch(data, "XGB", '18~20_ride')

   
    for n in range(1):
        acc_Alg2, rClf_Alg2, feat_Alg2, res_Alg2 = exp.sklearn_main(data, "LGB", 10000, '18~20_ride')
        resLst.append(res_Alg2)
        exp.jobStorage(rClf_Alg2, "Alg2_" + str(n), feat_Alg2)
    
    for n in range(1):
        acc_Alg3, rClf_Alg3, feat_Alg3, res_Alg3 = exp.sklearn_main(data, "XGB", 10000, '18~20_ride')
        resLst.append(res_Alg3)
        exp.jobStorage(rClf_Alg3, "Alg3_" + str(n), feat_Alg3)
    

    overallRes = np.transpose(resLst)
    calcRes = list()
    for oRes in overallRes:
        calcRes.append(int(np.round(np.average(oRes))))



    print("=======================================================================\n\n")

