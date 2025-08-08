import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

router = APIRouter()

# ✅ Load model with absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "disease_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# 🧾 Request data schema
class PatientData(BaseModel):
    age: float
    bmi: float
    bp: float
    glucose: float

# 💡 Health advice function
def generate_recommendation(score: float) -> str:
    if score > 0.8:
        return "High risk. Immediate medical consultation is advised."
    elif score > 0.5:
        return "Moderate risk. Lifestyle changes and monitoring recommended."
    else:
        return "Low risk. Maintain current healthy lifestyle."

# 🔮 Prediction route
@router.post("/predict")
async def predict_disease(data: PatientData):
    try:
        input_data = np.array([[data.age, data.bmi, data.bp, data.glucose]])
        prediction = model.predict(input_data)[0][0]
        recommendation = generate_recommendation(prediction)

        return {
            "risk_score": float(prediction),
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
