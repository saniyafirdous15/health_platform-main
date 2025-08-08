import tensorflow as tf
import numpy as np
import os

model_path = "ml/disease_model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}. Please train the model first.")

model = tf.keras.models.load_model(model_path)

def predict_disease(features: list) -> float:
    x = np.array([features])
    prediction = model.predict(x)
    return float(prediction[0][0])

