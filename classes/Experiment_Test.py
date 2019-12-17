import sys

CLASS_PATH = './classes'
sys.path.insert(0, CLASS_PATH)

import pandas as pd 
import numpy as np 
import pickle 
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from collections import Counter

from xgboost import XGBClassifier
import xgboost as xgb

import os
import csv
import copy

class Experiment_Test:
    cAllCases = 10000000000
    cAccThd = 95.0
    	
    def __init__(self, mAccThd, mType):
        self.cAccThd = float(mAccThd)
        self.type = mType
    
    def overall_Res_Multi_Class(self, mRes):
        weight_Df = pd.read_csv('./acc_weight.csv') 

        newRes = np.transpose(list(mRes.values()))
        newRes_ov = list()
        for nr in newRes:
            newRes_ov.append(np.round(np.average(nr)))    

        return np.array(newRes_ov)

    def writeAns(self, mInPath, mOutName, mAns):
        with open(mOutName, 'w') as fw:
            virList = pd.read_csv(mInPath, sep=',')

            virL = virList['domain'].values

            wr = csv.writer(fw)

            for vL, a in zip(virL, mAns):
                wrLst = list()
                wrLst.append(vL)
                wrLst.append(int(a))

                wr.writerow(wrLst)


    
    def jobLoader(self, mJob_Name):
        clf = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../classifier_' + self.type + '/' + mJob_Name + '_classifier.pkl'))
        #clf = xgb.Booster({'nthread': 15, 'objective': 'multi:softmax', 'num_class': 20})
        #clf.load_model('./classifier_' + self.type + '/batch_Models.model')
        
        features = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../classifier_' + self.type + '/' + mJob_Name + '_feats.pkl'), 'rb'))
        return clf, features
 

