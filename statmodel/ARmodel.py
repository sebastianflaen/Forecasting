import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AR, ARResults
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

df = pd.read_csv('../statmodel/Data/uspopulation.csv', index_col='DATE',parse_dates=True)
df.index.freq = 'MS'
df.plot()
plt.show()

warnings.filterwarnings('ignore')

train = df.iloc[:84]
test = df.iloc[84:]

model = AR(train['PopEst'])
AR1fit = model.fit(maxlag=1)
AR1fit.params #prints konst and estimate

start = len(train)
end = len(train) + len(test) -1
prediction1 = AR1fit.predict(start=start,end=end)
prediction1 = prediction1.rename('AR(1) predicitons')

test.plot(figsize=(12,8),legend=True)
prediction1.plot(legend=True)
plt.show()

model2 = AR(train['PopEst'])
AR2fit = model2.fit(maxlag=2)
predictions2 = AR2fit.predict(start,end)

predictions2 = predictions2.rename('AR(2) pred: ')

test.plot(figsize=(12,8),legend=True)
prediction1.plot(legend=True)
predictions2.plot(legend=True)
plt.show()
model3 = AR(train['PopEst'])
ARfit = model3.fit(ic='t-stat') #information criterion to choose optimal lag length.many different
labels = ['AR1','AR2','AR8']
predictions8 = ARfit.predict(start,end)
predictions8 = predictions8.rename('AR(8)pred:')
preds =[prediction1,predictions2,predictions8]
test.plot(figsize=(12,8),legend=True)
prediction1.plot(legend=True)
predictions2.plot(legend=True)
predictions8.plot(legend=True)
plt.show()
for i in range(3):
    error = mean_squared_error(test['PopEst'],preds[i])
    print(f'{labels[i]} MSE was :{error}')


#FORECASTING In the fuuuuture

model = AR(df['PopEst'])
ARfitted = model.fit()

forcasted_values = ARfitted.predict(start=len(df),end=len(df)+20).rename('FORECAST')
df['PopEst'].plot(legend=True)
forcasted_values.plot(legend=True)
plt.show()

#TEST for stationary: augmented dickey fuller test. Returns p value. reject if if low > 0.05
#Granger causality test. determine if one time series is useful in forecasting another -> causality.
df3 = pd.read_csv('../statmodel/Data/samples.csv',index_col=0,parse_dates=True)
df3.index.freq ='MS'
df3[['a','d']].plot()
plt.show()

#read the p-values. Very low p-values -> causality
print(grangercausalitytests(df3[['a','d']], maxlag=3))
print(grangercausalitytests(df3[['b','d']], maxlag=3))