from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from google import genai
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()

# Gemini client

client = genai.Client(
api_key=os.getenv("GEMINI_API_KEY")
)

# FastAPI app

app = FastAPI(title="Road Accident AI API")

# Enable CORS

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_methods=["*"],
allow_headers=["*"]
)

# =========================

# Model Path Fix

# =========================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "accident_risk_model.pkl")

model = joblib.load(model_path)


class AccidentInput(BaseModel):
    hour: int
    month: int
    day_of_week: int
    weather: int
    road_surface: int
    light_condition: int
    road_type: int
    speed_limit: int



@app.get("/")
def home():
    return {"message": "Road Accident AI Backend Running"}



@app.post("/predict")
def predict(data: AccidentInput):

features = np.array([[
    data.hour,
    data.month,
    data.day_of_week,
    data.weather,
    data.road_surface,
    data.light_condition,
    data.road_type,
    data.speed_limit
]])

prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0].max()

risk_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

risk_level = risk_map.get(prediction, "Medium")

prompt = f"""
```

A road accident prediction system detected:

Risk Level: {risk_level}
Confidence: {probability:.0%}
Time: {data.hour}:00
Weather Code: {data.weather}
Speed Limit: {data.speed_limit}

Write a short road safety warning in simple English.
"""

```
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

ai_message = response.text

return {
    "risk_level": risk_level,
    "confidence": f"{probability:.0%}",
    "ai_warning": ai_message
}

