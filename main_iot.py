from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

def prepare_input(data):
    input_dict = {col: 0 for col in columns}

    input_dict['Age'] = data['age']
    input_dict['HeartRate'] = data['hr']
    input_dict['SpO2'] = data['spo2']
    input_dict['Temperature'] = data['temp']
    input_dict[f"Activity_{data['activity']}"] = 1

    return pd.DataFrame([input_dict])

@app.post("/predict")
def predict(data: dict):
    df = prepare_input(data)
    result = model.predict(df)[0]

    return {"state": int(result)}