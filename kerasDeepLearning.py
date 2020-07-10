import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#y =mx + b+ noise
m = 2
b =3
x =np.linspace(0,50,100)
np.random.seed(101)
noise = np.random.normal(loc=1,scale=4,size=len(x))
y = 2*x + b +noise
print(y)
plt.plot(x,y,'x')
plt.show()

model = Sequential()
model.add(Dense(4,input_dim=1,activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse',optimizer='adam')

model.fit(x,y,epochs=200)