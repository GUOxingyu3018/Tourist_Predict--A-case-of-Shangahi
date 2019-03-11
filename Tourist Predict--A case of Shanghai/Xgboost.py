import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
from xgboost import plot_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

def trainandTest(X_train, Y_train, X_test,Y_test):


    #model = xgb.XGBRegressor(max_depth=5, learning_rate=0.3, n_estimators=100, silent=True, objective='reg:gamma')
    model = xgb.XGBClassifier(
                 learning_rate =0.1,
                 n_estimators=3000,
                 max_depth=4,
                 min_child_weight=1,
                 gamma=0,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 objective= 'binary:logistic',
                 nthread=4,
                 scale_pos_weight=1,
                 seed=27)
    model.fit(X_train, Y_train)
    ans = model.predict(X_test)

    r2 = r2_score(np.array(Y_test),np.array(ans))
    print(r2)


    plt.plot(np.array(Y_test),color = 'red',label = 'tourist_Number')
    plt.plot(np.array(ans),color = 'green',label = 'predValue')
    plt.legend()
    plt.show()





if __name__ == '__main__':
    origin_Data = pd.read_csv(r'Data/origin_Data.csv',encoding='ISO-8859-1',usecols = [1,2,3,4])
    X_train,X_test = np.array(origin_Data.ix[:60,:-1]),np.array(origin_Data.ix[:,:-1])
    Y_train,Y_test = np.array(origin_Data.ix[:60,-1]),np.array(origin_Data.ix[:,-1])

    #X_train, X_test, Y_train, Y_test = train_test_split(o_xList, o_yList, test_size=0.3, random_state=0)

    trainandTest(X_train, Y_train, X_test,Y_test)

