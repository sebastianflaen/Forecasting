# have to choose p,d,q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read pacf and acf. more realistic to use gridsearch
#pmdarima. AIC to compare

from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
#non-stationary
df1 = pd.read_csv('./statmodel/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
#stationary
df2 = pd.read_csv('./statmodel/Data/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df1.index.freq = 'MS'
df2.index.freq = 'D'

stepwise_fit = (auto_arima(df2['Births'],start_p=0,start_q=0,max_p=6,max_q=3,seasonal=False,trace=True))
print(stepwise_fit)
print(stepwise_fit.summary())

stepwise_fit2 =auto_arima(df1['Thousands of Passengers'],start_p=0,start_q=0,max_p=4,max_q=4,seasonal=True,trace=True, m = 12)
print(stepwise_fit2)