from statsmodels import * 
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot 
import statsmodels.api as sm

data = pd.read_excel(r'Data/')#读取excel数据
tourist_Number = data['旅游人数']

#数据平稳性检验，确定模型的D取值
def stationarity_Test(data):
    data_diff = data.diff()
    pyplot.plot(data, label = '原始数据', colors = 'black')
    pyplot.plot(data_diff, label = '一阶差分', colors = 'red')
    pyplot.legend()
    pyplot.rcParams['savefig.dpi'] = 300 #图片像素
    pyplot.rcParams['figure.dpi'] = 300 #分辨率
    pyplot.savefig(r'Pictures/%s'%(data) + '.png')
    pyplot.show()


#确定模型的P,Q最佳取值
def decide_PQ():   
    p = d = q = range(0, 4)    
    pdq = list(itertools.product(p, d, q))   
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
 
                results = mod.fit()
 
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
 
    results = mod.fit()
    print(results.summary().tables[1])
 
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()

    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
    pred_ci = pred.conf_int()


if __name__ == '__main__':
    stationarity_Test()