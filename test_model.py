import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


data = pd.read_csv('/Users/dazzle/Downloads/diabetes_prediction_dataset.csv')

# Same encoding as training
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
data['smoking_history'] = data['smoking_history'].astype('category').cat.codes


split = int(0.8 * len(data))
test = data.iloc[split:, :].reset_index(drop=True)

test_x = test.iloc[:, 0:-1]
test_y = test.iloc[:, -1].values


scaler = joblib.load("scaler.pkl")
test_x = scaler.transform(test_x)


model = load_model("diabetes_model_optimized.h5")
print("Model + Scaler loaded")


predictions = model.predict(test_x)
predicted_classes = (predictions > 0.5).astype(int).flatten()


test['Predicted_Diabetes'] = predicted_classes
test['Prediction_Probability'] = predictions.flatten()

accuracy = np.mean(predicted_classes == test_y)
print("✅ Test Accuracy:", accuracy)

output_file = "test_optimized_predictions.csv"
test.to_csv(output_file, index=False)
print(f"✅ Predictions saved to {output_file}")
