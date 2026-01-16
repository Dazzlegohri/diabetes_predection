import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load dataset
data = pd.read_csv('/Users/dazzle/Downloads/diabetes_dataset_2000_rows.csv')

# FIXED encoding (same as training)
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
smoking_map = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3}

data['gender'] = data['gender'].map(gender_map).fillna(0)
data['smoking_history'] = data['smoking_history'].map(smoking_map).fillna(0)

# Split X, y
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1].values

# Load scaler
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# Load model
model = load_model("diabetes_model_optimized.h5")
print("Model + Scaler loaded")

# Predict
predictions = model.predict(X_scaled)
predicted_classes = (predictions > 0.5).astype(int).flatten()

data['Predicted_Diabetes'] = predicted_classes
data['Prediction_Probability'] = predictions.flatten()

accuracy = np.mean(predicted_classes == y)
print("⚠️ Accuracy on FULL dataset:", accuracy)

data.to_csv("full_dataset_predictions_fixed.csv", index=False)
print("✅ Saved as full_dataset_predictions_fixed.csv")
