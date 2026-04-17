from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI()

# -------------------------------
# Enable CORS (IMPORTANT)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load ML model
# -------------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None
    print("Model loading failed:", e)

# -------------------------------
# OpenAI client (uses env variable)
# -------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# Request Schemas
# -------------------------------
class PredictionInput(BaseModel):
    distance: float
    carrier_rating: float
    weather_score: float

class ChatInput(BaseModel):
    message: str

# -------------------------------
# Routes
# -------------------------------

@app.get("/")
def home():
    return {"status": "API is running"}

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        features = np.array([[
            data.distance,
            data.carrier_rating,
            data.weather_score
        ]])

        prediction = model.predict(features)[0]

        return {
            "predicted_delay_days": round(float(prediction), 2),
            "risk_level": "High" if prediction > 3 else "Low"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# AI Chat Endpoint
# -------------------------------
@app.post("/chat")
def chat(data: ChatInput):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a logistics AI assistant. Help analyze shipment delays and suggest optimizations."
                },
                {
                    "role": "user",
                    "content": data.message
                }
            ]
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Health Check (for deployment)
# -------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}
