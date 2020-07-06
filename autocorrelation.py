import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings


from statsmodels.tsa.stattools import acovf, acf,pacf,pacf_yw,pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Non stat

df1 = pd.read_csv('./airline_passengers.csv',index_col='Month',parse_dates=True)
df1.index.freq = 'MS'

#df2 = pd.read_csv('statmodel/Data/DailyTotalFemaleBirths.csv', index_col='Data',parse_dates=True)
#df2.index.freq = 'D'
df2 = pd.read_csv('statmodel/Data/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df2.index.freq = 'D'
#warnings.filterwarnings('ignore')
df1.plot()
plt.show()

#non-stationary
plot_acf(df1, lags=40)


#stationary, ikkeno seasonality her
plot_acf(df2,lags=40)
plt.show()


plot_pacf(df2,lags=40, title="Partial Auto Correlation")
plt.show()