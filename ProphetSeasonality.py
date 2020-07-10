import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('./statmodel/Data/airline_passengers.csv')

#additive or multiplicative

df.columns = ['ds','y']
df['ds']=pd.to_datetime(df['ds'])

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(50,freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
plt.show()

fig = m.plot_components(forecast)
plt.show()

#Ser at det er multiplicative seasonality

from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)
plt.show()

#change seasonality to multiplicative.
s = Prophet(seasonality_mode='multiplicative')
s.fit(df)
future = s.make_future_dataframe(50,freq='MS')
forecast = s.predict(future)
fig = s.plot(forecast)
plt.show()
fig = m.plot_components(forecast)
plt.show()