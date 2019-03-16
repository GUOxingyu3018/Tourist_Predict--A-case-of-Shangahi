import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

#对数据进行格式处理，匹配模型数据结构要求
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values

	if dropnan:
		agg.dropna(inplace=True)
	return agg

#构建LSTM预测模型
def lstm_Model(data):

    #取2013-2017年数据为训练数据，2018年数据为预测数据
    x_train,x_test = data.ix[:60,:-1],data.ix[60:,:-1],
    y_train,y_test = data.ix[:60,-1],data.ix[60:,-1]


    # reshape input to be 3D [samples, timesteps, features]
    x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))


    #模型参数
    model = Sequential()
    model.add(LSTM(1000, input_shape=(x_train.shape[1], x_train.shape[2]),return_sequences=True))
    model.add(LSTM(units=750,return_sequences=True))
    model.add(LSTM(units=500,return_sequences=True))
    model.add(LSTM(units=250,return_sequences=True))
    model.add(LSTM(units=200,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')


    history = model.fit(x_train, y_train , batch_size=1,epochs=10, validation_data=(x_test, y_test), verbose=2, shuffle=False)

    predValue = model.predict(x_test)

    #对预测值于真实值的R2
    r2 = r2_score(np.array(y_test),np.array(predValue))
    print(r2)

    plt.title('上海旅游预测-LSTM r2={0}'.format(r2))
    plt.plot(np.array(y_test),color = 'red',label = 'tourist_Number')
    plt.plot(np.array(predValue),color = 'green',label = 'predValue')
    plt.legend()
    plt.show()


    return predValue

if __name__ == '__main__':
    idx = pd.period_range('1/1/2013','12/31/2017 ',freq='M')#时间索引

    origin_data = pd.read_excel(r'Data/建模数据.xlsx',sheet_name='优化模型',usecols=[0,1,2,3])

    x_std = StandardScaler().fit_transform(origin_data)#归一化

    data = series_to_supervised(x_std)
    lstm_Model(data)
    #tN_std = StandardScaler().fit_transform(pd.read_excel(r'Data/建模数据.xlsx',sheet_name='优化模型',usecols=[3]))#旅游人数归一化
    #pV = StandardScaler().inverse_transform(lstm_Model(data))#旅游人数还原
    #print(pV)
