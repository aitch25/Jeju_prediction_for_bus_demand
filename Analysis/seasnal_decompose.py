import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm
#series = [i**2.0 for i in range(1,100)]
data = pd.read_csv(sys.argv[1])

#result = seasonal_decompose(data['18~20_ride'], model='multiplicative', freq=1)
#result = seasonal_decompose(data['18~20_ride'], model='additive', freq=1)

for n in tqdm(range(1, 200000, 1000)):
    result = seasonal_decompose(data['18~20_ride'], model='additive', freq=n)
    result.plot()
    pyplot.savefig('./s_Figs/seasonality_' + str(n) + '.png')
    pyplot.show()
