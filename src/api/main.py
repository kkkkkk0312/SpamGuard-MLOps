from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

class Message(BaseModel):
    text: str

app = FastAPI()

model = None
vectorizer = None

@app.on_event("startup")
def load_model():
    global model, vectorizer
    with open("models/model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)

@app.post("/predict")
def predict(msg: Message):
    vec = vectorizer.transform([msg.text])
    prediction = model.predict(vec)[0]
    return {"result": "spam" if prediction else "ham"}
