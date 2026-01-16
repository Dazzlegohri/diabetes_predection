import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# Load data
full_data = pd.read_csv('/Users/dazzle/Downloads/diabetes_prediction_dataset.csv')

# Encode categorical columns (MINIMUM REQUIRED CHANGE)
full_data['gender'] = full_data['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
full_data['smoking_history'] = full_data['smoking_history'].astype('category').cat.codes

print("Dataset shape is : ", full_data.shape)

# Model parameters
hidden_units = 100
learning_rate = 0.01
hidden_layer_act = 'tanh'
output_layer_act = 'sigmoid'
no_epochs = 200

# Model
model = Sequential()
model.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))

sgd = optimizers.SGD(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# -------- SAME PATTERN: MANUAL 80–20 SPLIT --------
train_data_mark = int(0.8 * len(full_data))
print("Train mark : ", train_data_mark)

# Features (0 to 7) & target (8)
train_x = full_data.iloc[:train_data_mark, 0:8]
train_y = full_data.iloc[:train_data_mark, 8]

test_x = full_data.iloc[train_data_mark:, 0:8]
test_y = full_data.iloc[train_data_mark:, 8]

print("Train X:", train_x.shape)
print("Train Y:", train_y.shape)
print("Test X :", test_x.shape)
print("Test Y :", test_y.shape)

# Train on 80%
model.fit(
    train_x,
    train_y,
    epochs=no_epochs,
    batch_size=32,
    verbose=2
)

model.save("diabetes_model.h5")
print("Model saved")
trained_model = model.load("diabetes_model.h5")

# # Test on last 20%
# loss, acc = model.evaluate(test_x, test_y, verbose=0)
# print("Test Accuracy:", acc)
