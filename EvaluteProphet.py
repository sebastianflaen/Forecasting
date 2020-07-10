import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
df = pd.read_csv('./statmodel/Data/Miles_Traveled.csv')

# df.info får fram typer

df.columns=['ds','y']
df['ds'] = pd.to_datetime(df['ds'])

df.plot(x='ds',y='y')
plt.show()
#len(df)
train = df.iloc[:576]
test = df.iloc[576:]
m=Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=12,freq='MS')
forecast = m.predict(future)
ax = forecast.plot(x='ds',y='yhat',label='Predictions',legend = True)
test.plot(x='ds',y='y', label='True test data',legend=True,ax=ax,xlim=('2018-01-01','2019-01-01'))
plt.show()

#Evaluate goodness
predictions = forecast.iloc[-12:]['yhat']

RootMeanSquearedError = rmse(predictions,test['y'])
print(RootMeanSquearedError)
print(test.mean())

#Prophet diagnostic allows us to use cross validation
from fbprophet.diagnostics import cross_validation,performance_metrics
from fbprophet.plot import plot_cross_validation_metric

#Initial training period
initial = 5 * 365
initial = str(initial) + ' days'


#Period
period = 5*365
period = str(period) + ' days'

#Horizon how long forecats for each period
horizon = 365
horizon = str(horizon)+ ' days'

df_cv = cross_validation(m,initial=initial,period=period,horizon=horizon)
print(df_cv)

#len df_cv = 108
# print(performance_metrics(df_cv))

plot_cross_validation_metric(df_cv,metric='rmse')
plt.show()