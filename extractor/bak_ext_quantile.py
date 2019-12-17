import sys
from tqdm import tqdm
import pandas as pd

if __name__=="__main__":
    datPath = sys.argv[1]
    addPath = sys.argv[2]

    mainDat = pd.read_csv(datPath)
    addDat = pd.read_csv(addPath)
    
    feats = dict()
    feats['id'] = list()
    feats['routeCnt'] = list()

    idx = 10000
    routes = list(mainDat['bus_route_id'])[:idx]
    print('from 0 to ' + str(idx))
    for n, rou in zip(range(idx), tqdm(routes)):
        feats['id'].append(n)
        feats['routeCnt'].append(routes.count(rou))

    for idx in range(1, 40):
        idx = idx*10000
        print('from ' + str(idx) + ' to ' + str(idx+10000))
        routes = list(mainDat['bus_route_id'])[idx:idx+10000]
        for n, rou in zip(range(idx, idx+10000), tqdm(routes)):
            feats['id'].append(n)
            feats['routeCnt'].append(routes.count(rou))


    idx = 400000
    print('from ' + str(idx) + ' to ' + str(mainDat.shape[0]))
    routes = list(mainDat['bus_route_id'])[idx:]
    for n, rou in zip(range(idx, mainDat.shape[0]), tqdm(routes)):
        feats['id'].append(rou)
        feats['routeCnt'].append(routes.count(rou))


    new_df = pd.DataFrame.from_dict(feats)

    #print(new_df)
    new_df.to_csv('./DAT/3_stationCnt.csv', index=False)



    

        #print(mainDat['bus_route_id'][rou].describe())
    #print(addData)



