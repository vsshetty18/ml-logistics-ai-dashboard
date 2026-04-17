from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[ 
        data["distance"],
        data["carrier_rating"],
        data["weather_score"]
    ]])
    
    pred = model.predict(features)[0]
    return {"delay": float(pred)}
