import joblib 
import pandas as pd 

from fastapi import FastAPI,HTTPException,status
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware


class HeartDiseasesData(BaseModel):
    Age:int = Field(...,examples=[55],ge=0,le=100)
    Sex:str = Field(...,examples=["M"])
    ChestPainType:str = Field(...,examples=["ASY"])
    RestingBP:int = Field(...,examples=[140],ge=30,le=200)
    Cholesterol:int = Field(...,examples=[240],ge=100,le=600)
    FastingBS:int = Field(...,examples=[0],ge=0,le=1)
    RestingECG:str = Field(...,examples=["Normal"])
    MaxHR:int = Field(...,examples=[150],ge=40,le=220)
    ExerciseAngina:str = Field(...,examples=["N"])
    Oldpeak:float = Field(...,examples=[1.2])
    ST_Slope:str = Field(...,examples=["Flat"])

app = FastAPI(title="Heart Disease Prediction",version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_file_path = "model.pkl" 

model_pipeline = None

def load_model():
    global model_pipeline
    try:
        model_pipeline = joblib.load(model_file_path)
        print("model loaded successfully ")
    except Exception as e:
        print(f"Model loading error {e}")
        raise RuntimeError(
            "Could not load the model pipline"
        ) 

load_model()

@app.get("/")
def check_health():
    return{
        "status":"Online",
        "model loaded ":model_pipeline is not None
    }


@app.post("/predict")
def predict_disease(data:HeartDiseasesData):

    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavaliable "
        )

    try:

        input_dict = data.model_dump()

        input_df = pd.DataFrame([input_dict])

        prediction = model_pipeline.predict(input_df)

        return {
        "prediction": "Heart Disease" if prediction[0] == 1 else "No Heart Disease",
        "data": int(prediction[0])
    }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"prediction error : {str(e)}"
        )


new_data = pd.DataFrame([{
    "Age": 55,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 140,
    "Cholesterol": 240,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.2,
    "ST_Slope": "Flat"
}])

