from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("model_trained.pk4")
class StudentInput(BaseModel):
    Hours_Studied: float
    Previous_Scores: float
    Extracurricular_Activities: str
    Sleep_Hours: float
    Sample_Question_Papers_Practiced: int
@app.get("/")
def home():
    return {
        "Message": "Student Performance Prediction API"
    }
@app.post("/predict")
def prediction(data: StudentInput):

    input_df = pd.DataFrame({
        "Hours Studied": [data.Hours_Studied],
        "Previous Scores": [data.Previous_Scores],
        "Extracurricular Activities": [data.Extracurricular_Activities],
        "Sleep Hours": [data.Sleep_Hours],
        "Sample Question Papers Practiced": [data.Sample_Question_Papers_Practiced]
    })
    prediction = model.predict(input_df)
    return {
        "Predicted Performance Index": round(float(prediction[0]), 2)
    }