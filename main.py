from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class InputData(BaseModel):
    answers: list[int] 

model = joblib.load("modelo_estilo.pkl")

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.answers])
    return {"prediction": prediction[0]}
