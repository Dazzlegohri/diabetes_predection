import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

full_data=pd.read_csv('/Users/dazzle/Downloads/diabetes_prediction_dataset.csv')

full_data.diabetes.value_counts()

# full_data.bmi.hist()
# plt.show()

# print("Dataset loaded successfully")
# print("Shape:", full_data.shape)
# print(full_data.head())

hidden_units=100
learning_rate=0.01
hidden_layer_act='tanh'
output_layer_act='sigmoid'
no_epochs=200


model = Sequential()

model.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))


sgd=optimizers.SGD(lr=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])

full_data.head()

print("Dataset shape is : ", full_data.shape)

train_data_mark = int(0.8 * len(full_data))
print("Train mark : ", train_data_mark)

train_x = full_data.iloc[0:train_data_mark, 0:7]
print("Length of train_x : ", len(train_x))
train_x.head()

train_y = full_data.iloc[0:train_data_mark, 8]
print("Length of train_y : ", len(train_y))
train_y.head()


model.fit(full_data_x, full_data_y, epochs=no_epochs, batch_size=len(full_data),  verbose=2)    