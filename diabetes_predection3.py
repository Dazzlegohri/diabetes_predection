import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

full_data = pd.read_csv('/Users/dazzle/Downloads/diabetes_prediction_dataset.csv')

full_data['gender'] = full_data['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
full_data['smoking_history'] = full_data['smoking_history'].astype('category').cat.codes

print("Dataset shape:", full_data.shape)


split = int(0.8 * len(full_data))

train_x = full_data.iloc[:split, 0:8]
train_y = full_data.iloc[:split, 8]

test_x  = full_data.iloc[split:, 0:8]
test_y  = full_data.iloc[split:, 8]


scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x  = scaler.transform(test_x)


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(8,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)


history = model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=2
)


loss, acc = model.evaluate(test_x, test_y, verbose=0)
print("\n✅ Test Accuracy:", acc)
print("✅ Test Loss:", loss)


model.save("diabetes_model_optimized.h5")
print("💾 Optimized model saved as diabetes_model_optimized.h5")


import joblib
joblib.dump(scaler, "scaler.pkl")
print("💾 Scaler saved as scaler.pkl")
