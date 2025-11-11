from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("crop_yield_model.pkl")

app = FastAPI(title="Crop Yield Prediction")

class CropInput(BaseModel):
    Crop_Year: int
    Season: str
    Crop: str
    Area: float

@app.post("/predict")
def predict_yield(data: CropInput):
    input_data = np.array([[data.Crop_Year, data.Season, data.Crop, data.Area]])
    prediction = model.predict(input_data)
    return {"predicted_yield": float(prediction[0])}
