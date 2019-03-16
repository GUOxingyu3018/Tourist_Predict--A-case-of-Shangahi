import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
from xgboost import plot_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

def trainandTest(X_train, Y_train, X_test,Y_test):

    model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=8500, max_depth=5, min_child_weight=5, seed=0,
                             subsample=1, colsample_bytree=1, gamma=0, reg_alpha=0, reg_lambda=0)
    model.fit(X_train, Y_train)
    ans = model.predict(X_test)
    r2 = r2_score(np.array(Y_test),np.array(ans))
    print(ans)
   
    plt.title('上海旅游预测-xgboost r2={0}'.format(r2))
    plt.plot(np.array(Y_test),color = 'red',label = 'tourist_Number',linewidth=4)
    plt.plot(np.array(ans),color = 'green',label = 'predValue',linestyle="--")
    plt.legend()
    plt.savefig(r'Pictures/xgboost-SHLY_origin_Data.png')
    plt.show()
    '''
    cv_params = {'colsample_bytree': [0.1,0.2,0.3,0.4,0.5,0.6,0.7]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 360, 'max_depth': 0, 'min_child_weight': 0, 'seed': 0,
                    'subsample': 0.2, 'colsample_bytree': 0.1, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 0}

    model = xgb.XGBRegressor(**other_params)

    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
'''


'''
    最优参数
    BaiduIndex
    model = xgb.XGBRegressor(learning_rate=0.2, n_estimators=675, max_depth=5, min_child_weight=5, seed=1,
                             subsample=0.8, colsample_bytree=0.7, gamma=0, reg_alpha=0, reg_lambda=1)
    SHLY   
    model = xgb.XGBRegressor(learning_rate=0.2, n_estimators=900, max_depth=5, min_child_weight=5, seed=0,
                             subsample=0.6, colsample_bytree=0.5, gamma=0, reg_alpha=0, reg_lambda=0)
    SHLY_origin_Data
    
    model = xgb.XGBRegressor(learning_rate=0.2, n_estimators=8500, max_depth=5, min_child_weight=5, seed=0,
                             subsample=0.2, colsample_bytree=0.1, gamma=0, reg_alpha=0, reg_lambda=0)
'''
    
    






if __name__ == '__main__':
    origin_Data = pd.read_csv(r'Data/SHLY_origin_Data.csv',encoding='ISO-8859-1')
    data_std =StandardScaler().fit_transform(origin_Data)#归一化
    X_train,X_test = np.array(origin_Data)[:84,:-1],np.array(origin_Data)[84:,:-1]
    Y_train,Y_test = np.array(origin_Data)[:84,-1],np.array(origin_Data)[84:,-1]

    trainandTest(X_train, Y_train, X_test,Y_test)