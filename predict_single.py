import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("diabetes_model.h5")
print("Model loaded successfully\n")

# Helper functions for encoding
def encode_gender(g):
    g = g.lower()
    if g == "male":
        return 0
    elif g == "female":
        return 1
    else:
        return 2

def encode_smoking(s):
    s = s.lower()
    if s == "no info":
        return 0
    elif s == "never":
        return 1
    elif s == "former":
        return 2
    else:
        return 3

# Take user input
print("Enter patient details:\n")

gender = encode_gender(input("Gender (Male/Female/Other): "))
age = float(input("Age: "))
hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))
heart_disease = int(input("Heart Disease (0 = No, 1 = Yes): "))
smoking_history = encode_smoking(input("Smoking History (No Info/Never/Former/Current): "))
bmi = float(input("BMI: "))
hba1c = float(input("HbA1c Level: "))
glucose = float(input("Blood Glucose Level: "))

patient = np.array([[
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c,
    glucose
]])

probability = model.predict(patient)[0][0]

print("\n--- Prediction Result ---")
print(f"Diabetes Probability : {probability:.2f}")

if probability > 0.5:
    print("⚠️ Prediction: Diabetes Detected")
else:
    print("✅ Prediction: No Diabetes")
