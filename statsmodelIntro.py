#Timeseries have trends. Daownward/Upward/horis/stati
#Seasonality - repeating trends in periods
#Cyclical - trends with no set repetition s&p500
#lambda smooting parameter

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



from statsmodels.tsa.filters.hp_filter import hpfilter

df = pd.read_csv('macrodata.csv', index_col=0, parse_dates=True)

df['realgdp'].plot.line(figsize = (10,3), c = 'blue', lw = 5)
plt.show()

gdp_cycle,gdp_trend = hpfilter(df['realgdp'], lamb=1600)


gdp_trend.plot()
df['trend'] = gdp_trend

df[['trend','realgdp']]['2005-01-01':].plot(figsize = (12,5))


plt.show()
#ETS Models (Error-Trend-Seasonality)


airline = pd.read_csv('airline_passengers.csv', index_col="Month")
airline.dropna(inplace=True)
airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()
airline.plot()
plt.show()


#EWMA exp weighted moving average@

airline['EWMA-12'] = airline['Thousands of Passengers'].ewm(span=12).mean()

airline[['Thousands of Passengers', 'EWMA-12']].plot(figsize=(10,8))
plt.show()

#Holt, simple exp, same as ewma
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
holt = pd.read_csv('airline_passengers.csv', index_col="Month",parse_dates=True)
holt = holt.dropna()
holt.index.freq = 'MS'
span = 12
alpha = 2/(span+1)

model = SimpleExpSmoothing(holt['Thousands of Passengers'])
fitted_model = model.fit(smoothing_level=alpha, optimized=False)
holt['SE12'] = fitted_model.fittedvalues.shift(-1)
holt.plot()
plt.show()
#Holt, double exp, much better
from statsmodels.tsa.holtwinters import ExponentialSmoothing
holt['DES_add_12'] = ExponentialSmoothing(holt['Thousands of Passengers'],trend= 'mul').fit().fittedvalues.shift(-1)
#mul : multiplative, add: additativ
holt[['Thousands of Passengers','SE12','DES_add_12']].iloc[-24:].plot(figsize=(12,5))
plt.show()


#Holt, tripple exp, much better in forecasting. BEcause of seasonality
holt['TES_add_12'] = ExponentialSmoothing(holt['Thousands of Passengers'],trend= 'mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues()
holt[['Thousands of Passengers','DES_add_12','TES_add_12']].plot(figsize=(12,6))
plt.show()




