import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


from statsmodels.tsa.stattools import acovf, acf,pacf,pacf_yw,pacf_ols

#Non stat

df1 = pd.read_csv('./airline_passengers.csv',index_col='Month',parse_dates=True)
df1.index.freq = 'MS'

df2 = pd.read_csv()