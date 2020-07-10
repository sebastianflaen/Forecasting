import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('./statmodel/Data/BeerWineLiquor.csv')
#print(df.head())

#firmat ds y

df.columns = ['ds','y']
#print(df.head())

df['ds'] = pd.to_datetime((df['ds']))
#print(df.head())
print(df.info())

m = Prophet()
m.fit(df)

#Placeholder to hold our future predictions. Holder av 24 y verdier.
#df.index.freq
future = m.make_future_dataframe(periods=24,freq='MS')

print(future)

forecast = m.predict(future)

forecast[['ds','yhat_lower','yhat_upper','yhat']].tail(12)

m.plot(forecast)

plt.show()
#plt.xlim('2014-01-01','2021-01-01')
m.plot_components(forecast) #viser trend og seasonality.
plt.show()
