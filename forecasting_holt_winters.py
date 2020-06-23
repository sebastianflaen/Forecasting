import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS' #månedlig frekvens.

train_data = df.iloc[:108] # Goes up to but not including 108
test_data = df.iloc[108:]


fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

test_predictions = fitted_model.forecast(36).rename('HW Forecast')

train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8));
test_predictions.plot(legend=True,label='PREDICTION');


plt.show()
train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION',xlim=['1958-01-01','1961-01-01']);
plt.show()


#Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

print(test_data.describe()) #sammenligningsgrunnlag

print(mean_absolute_error(test_data,test_predictions))

print(mean_squared_error(test_data,test_predictions))

print(np.sqrt(mean_squared_error(test_data,test_predictions))) #root mean abs dev?

#har nå sammenlignet med test_data og_ test predicition.

#ønsker nå å faktisk forecaste inn iframtiden. Data vi ikke har enda

final_model = ExponentialSmoothing(df['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
forecast_predictions = final_model.forecast(36)

df['Thousands of Passengers'].plot(figsize=(12,8))
forecast_predictions = final_model.forecast(36)
df['Thousands of Passengers'].plot(figsize=(12,8))
forecast_predictions.plot();
plt.show()


#stationarity: no seasonality or trends
