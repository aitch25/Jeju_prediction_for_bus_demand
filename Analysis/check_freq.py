import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm
#series = [i**2.0 for i in range(1,100)]

def datPrep(mData, mDate):
    keyA = mData['bus_route_id']
    keyB = mData['station_code']

    newHist = pd.DataFrame({'NewKey' : keyA*10000 + keyB})
    newHist = newHist.drop_duplicates()

    mData['NewKey'] = keyA*10000 + keyB

    #print(mData[mData['date'] == '2019-09-01'])
    newDat = mData[mData['date'] == mDate][['NewKey', '18~20_ride']]

    newHist = pd.merge(newHist, newDat, how='left', left_on='NewKey', right_on='NewKey')
    #print(newHist.fillna(0))
    return newHist.fillna(0)
    


if __name__=="__main__":
    data = pd.read_csv(sys.argv[1])

    #y1 = datPrep(data, '2019-09-01')
    #y2 = datPrep(data, '2019-09-02')
    #y3 = datPrep(data, '2019-09-03')
    
    plots = 29
    for plot in tqdm(range(1, plots)):

        date_p0 = str('2019-09-%02d' % plot)
        y1 = datPrep(data, date_p0)
        plt.subplot(2, 1, 1)
        plt.plot(y1, label=date_p0)
        plt.legend(loc='upper right')
        plt.axis([0, len(y1), 0, 300])
        plt.show()

        date_p1 = str('2019-09-%02d' % (plot+1))
        y2 = datPrep(data, date_p1)
        plt.subplot(2, 1, 2)
        plt.plot(y2, label=date_p1)
        plt.legend(loc='upper right')
        plt.axis([0, len(y2), 0, 300])
        plt.show()
    
        plt.savefig('./freq_figs/freqs_' + str(plot) + '.png')
        plt.clf()
