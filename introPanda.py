import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


df = pd.read_csv('a.csv')

df['A'].plot.line(figsize = (10,3), ls = ':', c = 'red', lw = 5)
plt.show()


my_date=(1,2,13,30,15)


np.array(['2020-03-15','2020-03-16','2020-03-15'], dtype='datetime64')

ar = np.arange('2018-06-01','2018-06-23',7,dtype = 'datetime64[D]')

print(ar)


pr = pd.date_range('2020-01-01', periods=7,freq='D')


#Statsmodel library:

