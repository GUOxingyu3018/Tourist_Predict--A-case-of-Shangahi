from statsmodels import * 
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import statsmodels.api as sm
import time
import seaborn as sns
import itertools
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime

data = pd.read_excel(r'Data/上海旅游人数--上海旅游（百度指数-区分来源）.xlsx', parse_dates=[0],index_col='时间')#读取excel数据
x = pd.read_excel(r'Data/自变量.xlsx', parse_dates=[0],index_col='时间')
data.index =  pd.to_datetime(data.index)
x.index = pd.to_datetime(x.index)

tourist_Number = data['旅游人数']
y_Train = tourist_Number[:85] #模型训练Y
y_Test = tourist_Number[85:] #检测Y

x_Train = x[:85]

print(data.head())

#绘图显示中文及设置分辨率
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率


#数据平稳性检验，确定模型的D取值
#图片显示旅游人数无需差分处理
def stationarity_Test(data):
    data_diff = data.diff()
    plt.plot(data, label = '上海旅游人数', color = 'black')
    plt.plot(data_diff, label = '上海旅游人数一阶差分', color = 'red')
    plt.legend()   
    plt.savefig(r'Pictures/差分图.png')
    plt.show()


#通过ACF与PACF图分析P，Q取值
def ACF_PACF(data):
    fig = plt.figure(figsize=(12,8))

    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=20,ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout();

    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=20, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout();
    plt.savefig(r'Pictures/ACF_PACF.png')
    plt.show()


#绘制散点图，判断P,D
def scatter_diagram(data):
    lags=9
    ncols=3
    nrows=int(np.ceil(lags/ncols))

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows))

    for ax, lag in zip(axes.flat, np.arange(1,lags+1, 1)):
        lag_str = 't-{}'.format(lag)
        X = (pd.concat([data, data.shift(-lag)], axis=1,
                    keys=['y'] + [lag_str]).dropna())

        X.plot(ax=ax, kind='scatter', y='y', x=lag_str);
        corr = X.corr().as_matrix()[0][1]
        ax.set_ylabel('Original')
        ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
        ax.set_aspect('equal');
        sns.despine();

    fig.tight_layout();
    plt.savefig(r'Pictures/散点图.png')
    plt.show()


#确定模型的P,Q最佳取值
def decide_PQ(y_Train,x_Train):   
    arima401 = sm.tsa.SARIMAX(y_Train,x_Train, order=(4,0,1))
    model_results = arima401.fit()

    p_min = 0
    d_min = 0
    q_min = 0
    p_max = 4
    d_max = 0
    q_max = 4

    results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

    for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
        if p==0 and d==0 and q==0:
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue
    
        try:
            model = sm.tsa.SARIMAX(y_Train,x_Train, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
            results = model.fit()
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
        except:
            continue
    results_bic = results_bic[results_bic.columns].astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 );
    ax.set_title('BIC');
    plt.savefig(r'Pictures/BIC.png')
    plt.show()


#构建ARIMA(0,0,1)预测模型
def pred():
    arima001 = sm.tsa.SARIMAX(y_Train,x_Train, order=(0,0,1))
    model = arima001.fit()
    
    pred = model.predict()

    print(pred)

#时间序列聚类，减少共线性
def linear_Clustering():
    extraction_settings = ComprehensiveFCParameters()
    X = extract_features(x, 
                         column_id='id',
                        column_sort='time', 
                        default_fc_parameters=extract_settings, 
                        impute_function=impute)

if __name__ == '__main__':
    #stationarity_Test(tourist_Number)
    #ACF_PACF(tourist_Number)
    #scatter_diagram(tourist_Number)
    #decide_PQ(y_Train,x_Train)  
    #最终确定ARIMA（0，0，1）
    pred()
