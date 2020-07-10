import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('./statmodel/Data/HospitalityEmployees.csv')
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
df.plot(x='ds',y='y')
plt.show()

m = Prophet()

m.fit(df)
future = m.make_future_dataframe(periods=12,freq='MS')
forecast =m.predict(future)


#shows us where the trends changes
from fbprophet.plot import add_changepoints_to_plot
fig = m.plot((forecast))
a = add_changepoints_to_plot(fig.gca(),m,forecast)
plt.show()