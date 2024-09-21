from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing. You can specify specific origins as needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# loading the saved model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: ModelInput) -> Dict[str, str]:
    input_list = [
        input_parameters.Pregnancies,
        input_parameters.Glucose,
        input_parameters.BloodPressure,
        input_parameters.SkinThickness,
        input_parameters.Insulin,
        input_parameters.BMI,
        input_parameters.DiabetesPedigreeFunction,
        input_parameters.Age
    ]
    prediction = diabetes_model.predict([input_list])
    if prediction[0] == 0:
        return {"result": "The Person is not Diabetic"}
    else:
        return {"result": "The Person is Diabetic"}
