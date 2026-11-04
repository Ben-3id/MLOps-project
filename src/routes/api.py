from fastapi import APIRouter , Request
from ..utilities.utilites import Preprocess
from ..settings import load_config
import pandas as pd
import logging 

logger = logging.getLogger("uvicorn") 


from pydantic import BaseModel
class StudentInput(BaseModel):
    Hours_Studied: float
    Previous_Scores: float
    Extracurricular_Activities: str
    Sleep_Hours: float
    Sample_Question_Papers_Practiced: int

api_route = APIRouter()

cfg = load_config()

@api_route.get("/")
def welcome():
    return "Hello from lobby"


@api_route.post("/predict")
def predict(input:StudentInput , req:Request):

    data = input.model_dump()
    df = pd.DataFrame([data])

    processor = Preprocess(cfg)
    example = processor.transform(df)
    prediction = req.app.model.predict(example)
    return {
        "prediction": prediction.tolist()[0]
    }


@api_route.post("/get-model")
def get_model(model_id:str , model_name:str, req:Request):

    model_path = req.app.mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{model_id}/{model_name}"
            )
    try:
        req.app.model = req.app.mlflow.sklearn.load_model(model_path)
        return "Successful"

    except: return "Can't find Model"