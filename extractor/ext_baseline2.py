import os
#os.chdir("참가자분들 각각의 데이터 경로 입력")

import numpy as np #데이터 처리
import pandas as pd #데이터 처리
import warnings
warnings.filterwarnings('ignore')
from collections import Counter # count 용도

import geopy.distance #거리 계산해주는 패키지 사용

import random #데이터 샘플링
from sklearn.model_selection import GridSearchCV #모델링
from sklearn.ensemble import RandomForestRegressor #모델링

import lightgbm as lgb
import xgboost as xgb


train = pd.read_csv("./DAT/origin/train.csv")
test = pd.read_csv("./DAT/origin/test.csv")

train['date2'] = pd.to_datetime(train['date'])

train['weekday'] = train['date2'].dt.weekday

train = pd.get_dummies(train,columns=['weekday'])

test['date2'] = pd.to_datetime(test['date'])
test['weekday'] = test['date2'].dt.weekday
test = pd.get_dummies(test,columns=['weekday'])

del train['date2']
del test['date2']
train['in_out'].value_counts()

train['in_out'] = train['in_out'].map({'시내':0,'시외':1})
test['in_out'] = test['in_out'].map({'시내':0,'시외':1})

train['68a']=train['6~7_ride']+train['7~8_ride'] # 6 ~ 8시 승차인원
train['79a']=train['7~8_ride']+train['8~9_ride'] 
train['810a']=train['8~9_ride']+train['9~10_ride']
train['911a']=train['9~10_ride']+train['10~11_ride']
train['1012a']=train['10~11_ride']+train['11~12_ride']

train['68b']=train['6~7_takeoff']+train['7~8_takeoff'] # 6 ~ 8시 하차인원
train['79b']=train['7~8_takeoff']+train['8~9_takeoff'] # 6 ~ 8시 하차인원
train['810b']=train['8~9_takeoff']+train['9~10_takeoff']
train['911b']=train['9~10_takeoff']+train['10~11_takeoff']
train['1012b']=train['10~11_takeoff']+train['11~12_takeoff']

#train22=train[['68a','810a','1012a','68b','810b','1012b','18~20_ride']]

#cor=train22.corr()


test['68a']=test['6~7_ride']+test['7~8_ride']
test['79a']=test['7~8_ride']+test['8~9_ride']
test['810a']=test['8~9_ride']+test['9~10_ride']
test['911a']=test['9~10_ride']+test['10~11_ride']
test['1012a']=test['10~11_ride']+test['11~12_ride']

test['68b']=test['6~7_takeoff']+test['7~8_takeoff']
test['79b']=test['7~8_takeoff']+test['8~9_takeoff']
test['810b']=test['8~9_takeoff']+test['9~10_takeoff']
test['911b']=test['9~10_takeoff']+test['10~11_takeoff']
test['1012b']=test['10~11_takeoff']+test['11~12_takeoff']

# 해당 주요 장소의 임의 지역 위도, 경도

jeju=(33.51411, 126.52969) # 제주 측정소 근처
gosan=(33.29382, 126.16283) #고산 측정소 근처
seongsan=(33.38677, 126.8802) #성산 측정소 근처
po=(33.24616, 126.5653) #서귀포 측정소 근처


#정류장의 위치만 확인하기 위해 groupby를 실행함
data=train[['latitude','longitude','station_name']].drop_duplicates(keep='first')

data2=data.groupby(['station_name'])['latitude','longitude'].mean()

data2.to_csv("folium.csv")

data2=pd.read_csv("folium.csv")


t1 = [geopy.distance.vincenty( (i,j), jeju).km for i,j in list( zip( train['latitude'],train['longitude'] )) ]
t2 = [geopy.distance.vincenty( (i,j), gosan).km for i,j in list( zip( train['latitude'],train['longitude'] )) ]
t3 = [geopy.distance.vincenty( (i,j), seongsan).km for i,j in list( zip( train['latitude'],train['longitude'] )) ]
t4 = [geopy.distance.vincenty( (i,j), po).km for i,j in list( zip( train['latitude'],train['longitude'] )) ]

train['dis_jeju']=t1
train['dis_gosan']=t2
train['dis_seongsan']=t3
train['dis_po']=t4

total=pd.DataFrame( list(zip( t1,t2,t3,t4)),columns=['jeju','gosan','seongsan','po'] )
train['dist_name'] = total.apply(lambda x: x.argmin(), axis=1)

data22=train[['station_name','latitude','longitude','dist_name']].drop_duplicates(keep='first')

Counter(data22['dist_name'])

t1 = [geopy.distance.vincenty( (i,j), jeju).km for i,j in list( zip( test['latitude'],test['longitude'] )) ]
t2 = [geopy.distance.vincenty( (i,j), gosan).km for i,j in list( zip( test['latitude'],test['longitude'] )) ]
t3 = [geopy.distance.vincenty( (i,j), seongsan).km for i,j in list( zip( test['latitude'],test['longitude'] )) ]
t4 = [geopy.distance.vincenty( (i,j), po).km for i,j in list( zip( test['latitude'],test['longitude'] )) ]

test['dis_jeju']=t1
test['dis_gosan']=t2
test['dis_seongsan']=t3
test['dis_po']=t4

total=pd.DataFrame( list(zip( t1,t2,t3,t4)),columns=['jeju','gosan','seongsan','po'] )
test['dist_name'] = total.apply(lambda x: x.argmin(), axis=1)


#데이터 불러오기
raining=pd.read_csv("./DAT/origin/weather_kma.csv",engine='python')

#외부데이터에서 나오는 지점명들을 변경
raining['dist_name'] = [ str(i) for i in raining['dist_name']]

raining['dist_name'] = ['jeju' if i=='184' else i for i in raining['dist_name'] ]  # 위도 : 33.51411 경도 : 126.52969
raining['dist_name'] = ['gosan' if i=='185' else i for i in raining['dist_name'] ]  # 위도 : 33.29382 경도 : 126.16283
raining['dist_name'] = ['seongsan' if i=='188' else i for i in raining['dist_name'] ]  # 위도 : 33.38677 경도 : 126.8802
raining['dist_name'] = ['po' if i=='189' else i for i in raining['dist_name'] ]  # 위도 : 33.24616 경도 : 126.5653

raining.head()

raining['time'] = [ int( i.split(' ')[1].split(':')[0] ) for i in raining['date']] 

raining['date'] = [ i.split(' ')[0] for i in raining['date'] ] 

# 실제 측정 데이터이기 때문에, 12시 이전의 시간대만 사용
rain2 = raining[ (raining['time']>=6) & (raining['time']<12)  ]


rain3 = rain2.groupby(['dist_name', 'date'])[['temp', 'rain']].mean()

#rain3.to_csv("rain3.csv")
#rain3=pd.read_csv("rain3.csv")

# train, test의 변수명과 통일시키고, NaN의 값은 0.0000으로 변경
#rain3 = rain3.rename(columns={"date":"date","pos":"dist_name"})
rain3 = rain3.fillna(0.00000)


train2 = pd.merge(train, rain3, how='left', on=['dist_name', 'date'])
test2 = pd.merge(test, rain3, how='left', on=['dist_name', 'date'])


train2 = pd.get_dummies(train2,columns=['dist_name'])
test2 = pd.get_dummies(test2,columns=['dist_name'])

print(train2.shape, test2.shape, train.shape, test.shape)

#input_var=['id', 'in_out','latitude', 'longitude', '68a', '79a', '810a', '911a', '1012a', '68b', '79b', '810b', '911b', '1012b', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'dis_jeju', 'dis_gosan','dis_seongsan', 'dis_po','temp', 'rain', 'dist_name_gosan', 'dist_name_jeju','dist_name_po', 'dist_name_seongsan']

input_var=['id', 'in_out', '68a', '79a', '810a', '911a', '1012a', '68b', '79b', '810b', '911b', '1012b', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'dis_jeju', 'dis_gosan','dis_seongsan', 'dis_po','temp', 'rain', 'dist_name_gosan', 'dist_name_jeju','dist_name_po', 'dist_name_seongsan']
#input_var=['id', '68a', '79a', '810a', '911a', '1012a', '68b', '79b', '810b', '911b', '1012b',
#           'dis_jeju', 'dis_gosan','dis_seongsan', 'dis_po','temp', 'rain', 
#           'dist_name_gosan', 'dist_name_jeju','dist_name_po', 'dist_name_seongsan']


target=['18~20_ride']


X_train=train2[input_var]
random.seed(1217) #동일한 샘플링하기 위한 시드번호
#train_list=random.sample(list(range(X_train.shape[0])), int(round(X_train.shape[0]*0.1,0)) )

X_train=train2[input_var]
#X_train=X_train.iloc[train_list,:]
y_train=train2[target]
#y_train=y_train.iloc[train_list,:]

X_test=test2[input_var]

print(X_train.shape, y_train.shape)



### for LGBM #####################
#param_grid = {
#    'max_depth': [2, 4, 8, 16, 32, 48, 64],
#}
#
#lgbm = lgb.LGBMRegressor(n_jobs=15, random_state=100)
#xgbr = xgb.XGBRegressor(n_jobs=15, random_state=100, predictor='gpu_predictor')
## Instantiate the grid search model
##grid_search = GridSearchCV(estimator = lgbm, param_grid = param_grid, n_jobs=15) # GridSearchCV를 정의한다.
#grid_search = GridSearchCV(estimator = xgbr, param_grid = param_grid) # GridSearchCV를 정의한다.
#
#grid_search.fit(X_train, y_train)
#
#print(grid_search.best_params_) #학습 이후 최적의 paramter를 출력

#해당 코드 실행시간 2분 ~ 3분 소요

#전체 데이터로 적용
X_train=train2[input_var]
y_train=train2[target]

X_test=test2[input_var].fillna(0)

X_train.to_csv('./DAT/TrainDat/02_baseline2.csv', index=False)
X_test.to_csv('./DAT/TestDat/02_baseline2.csv', index=False)
exit()

#rf = RandomForestRegressor(max_features=3,min_samples_leaf=2,min_samples_split=2,n_estimators=500,random_state=1217, n_jobs=15)

rf.fit(X_train, y_train) #학습 

test['18~20_ride'] = rf.predict(X_test) #예측값 생성 후, test['18~20_ride']에 집어 넣는다.

test[['id','18~20_ride']].to_csv("dacon_base_middle.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 내보낸다

