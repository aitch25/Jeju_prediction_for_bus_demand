import os
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
        
            #print('This!!!', mFeatLst.split('/'))
            feats_forUse = feats_forUse[['date', 'time', key]]

        else:
            print("Merge:", d)
            forMerge = pd.read_csv(dirPath)
            forMerge = forMerge[key]
            #forMerge = forMerge.dropna(axis='id')

            feats_forUse = pd.merge(feats_forUse, forMerge, left_index=True, right_index=True)
            print(feats_forUse.shape)


    return feats_forUse


if __name__=="__main__":

    dirs = os.listdir(sys.argv[1])

    for d in dirs:
        with open(sys.argv[1][:-1] + '_prep/' + d, 'w') as wr:
            outLine = str('date,time,' + (sys.argv[1]).split('/')[-2] + '\n')
            wr.write(outLine)

    for d in dirs:
        outLine = ''
        with open(sys.argv[1] + d, 'r') as files:
            lines = files.readlines()
            
            date_format = '2019-%02d-%02d'
            mon = 9
            for line in lines[1:-1]:
                if not 'Start' in line:
                    line = line.replace(' ', '')
                    line = line.replace('\n', '')
                    line = line.split(',')

                    if (int(line[1]) > 400) and (int(line[1]) < 1300):
                        #time_format = str('%02d:00:00')line[1] / 100
                        

                        outLine = str(date_format % (mon, int(line[0]))) + ','
                        outLine = outLine + line[1] + ','
                        outLine = outLine + line[2] + '\n'

                    elif (int(line[1]) > 1600) and (int(line[1]) < 2100):
                        outLine = str(date_format % (mon, int(line[0]))) + ','
                        outLine = outLine + line[1] + ','
                        outLine = outLine + line[2] + '\n'

                    else: continue

                    with open(sys.argv[1][:-1] + '_prep/' + d, 'a') as wr:
                        wr.write(outLine)


                else: 
                    mon = mon + 1


    head = list(['date', 'time'])
    for d in range(len(dirs)):
        head.append(sys.argv[1].split('/')[-2] + '_' + str(d))
        

    
    merged = selectFeature(sys.argv[1][:-1] + '_prep/')

    merged.index.name = 'id'
    #print(sys.argv[1][:-1] + '_merged/')
    #exit()
    print(merged)
    merged.to_csv(sys.argv[1] + '../merged/' + sys.argv[1].split('/')[-2] + '.csv', header=head, index=True)


