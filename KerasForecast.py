import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

df = pd.read_csv('./statmodel/Data/Alcohol_Sales.csv', index_col='DATE',parse_dates=True)
df.index.freq = 'MS'
df.columns = ['Sales']
df.plot()
plt.show()
result = seasonal_decompose(df['Sales'])
result.plot()
plt.show()

train = df.iloc[:313]
test = df.iloc[313:]
print(len(test))
scaler = MinMaxScaler()
scaler.fit(train) #finds the ma vale on the training data set
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

n_input = 2
n_features = 1
generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=1)
X,y = generator[0]
