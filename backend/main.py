# FastAPI is our web framework
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# For data validation
from pydantic import BaseModel

# For ML model
import joblib
import numpy as np
import pandas as pd

# For Gemini AI
import google.generativeai as genai

# For environment variables
import os
from dotenv import load_dotenv

# Load secret keys from .env file
load_dotenv()

print("All imports successful!")

# Create the app
app = FastAPI(
    title="Road Accident Risk Predictor API",
    description="Predicts road accident risk for Indian states",
    version="1.0.0"
)

# This allows frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load your saved ML model
try:
    model = joblib.load('../models/accident_risk_model.pkl')
    print(" ML Model loaded!")
except:
    print(" Model not found - check path")

# Load master dataset
try:
    master_df = pd.read_csv(
        '../data/processed/master_dataset.csv'
    )
    print(" Master dataset loaded!")
except:
    print(" Dataset not found - check path")

# Setup Gemini AI
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)
gemini = genai.GenerativeModel('gemini-1.5-flash')
print(" Gemini AI ready!")

# This tells FastAPI what input to expect
# from the frontend
class StateInput(BaseModel):
    state_name: str
    avg_killed: float
    avg_injured: float
    fine_accidents: float
    rain_accidents: float
    fog_accidents: float
    dry_road_accidents: float
    wet_road_accidents: float
    pothole_accidents: float

    # Route 1 - Home page check
@app.get("/")
def home():
    return {
        "message": "Road Accident Risk API is running!",
        "status": "active"
    }

# Route 2 - Get all states list
@app.get("/states")
def get_states():
    states = master_df['State'].tolist()
    return {"states": states}

# Route 3 - Get specific state data
@app.get("/state/{state_name}")
def get_state_data(state_name: str):
    state_data = master_df[
        master_df['State'] == state_name
    ]
    if state_data.empty:
        return {"error": "State not found"}

    data = state_data.iloc[0]
    return {
        "state": state_name,
        "avg_killed": round(
            float(data['Avg_Killed']), 0
        ),
        "avg_injured": round(
            float(data['Avg_Injured']), 0
        ),
        "risk_level": data['Risk_Level']
    }

# Route 4 - Main prediction endpoint
@app.post("/predict")
def predict_risk(data: StateInput):

    # Prepare features for ML model
    features = np.array([[
        data.avg_killed,
        data.avg_injured,
        data.fine_accidents,
        data.rain_accidents,
        data.fog_accidents,
        data.dry_road_accidents,
        data.wet_road_accidents,
        data.pothole_accidents
    ]])

    # Get prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = round(float(probability.max()) * 100, 1)

    # Color code for risk
    color_map = {
        'High'  : '🔴',
        'Medium': '🟡',
        'Low'   : '🟢'
    }
    risk_icon = color_map.get(prediction, '🟡')

    # Generate AI explanation using Gemini
    prompt = f"""
    You are a road safety expert in India.
    
    State: {data.state_name}
    Risk Level: {prediction}
    Average people killed per year: {int(data.avg_killed):,}
    Average people injured per year: {int(data.avg_injured):,}
    Accidents in fine weather: {int(data.fine_accidents):,}
    Accidents in rain: {int(data.rain_accidents):,}
    
    Write a 3 sentence safety advisory for this state.
    Be specific, helpful and easy to understand.
    Write as if advising the state traffic police.
    """

    try:
        response = gemini.generate_content(prompt)
        ai_advice = response.text
    except:
        ai_advice = (
            f"{data.state_name} shows {prediction} risk. "
            "Immediate road safety measures recommended."
        )

    return {
        "state"     : data.state_name,
        "risk_level": f"{risk_icon} {prediction}",
        "confidence": f"{confidence}%",
        "avg_killed": int(data.avg_killed),
        "avg_injured": int(data.avg_injured),
        "ai_advice" : ai_advice
    }

# Route 5 - Get all states with risk levels
@app.get("/all-risks")
def get_all_risks():
    result = []
    for _, row in master_df.iterrows():
        result.append({
            "state"     : row['State'],
            "risk_level": row['Risk_Level'],
            "avg_killed": round(row['Avg_Killed'], 0)
        })
    return {"data": result}
