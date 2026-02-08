import fastapi
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from src.pipeline.train_pipeline import model_pipeline
from mlflow import MlflowClient
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

app = fastapi.FastAPI()

class PatientData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(default=0)

    Age: int = Field(..., gt=0, lt=120)
    BP: int = Field(..., description="Resting blood pressure")
    Cholesterol: int
    Max_HR: int = Field(..., alias="Max HR")
    ST_depression: float = Field(..., alias="ST depression")
    Number_of_vessels_fluro: Literal[0, 1, 2, 3] = Field(..., alias="Number of vessels fluro")

    Sex: Literal[0, 1]
    Chest_pain_type: Literal[1, 2, 3, 4] = Field(..., alias="Chest pain type")
    FBS_over_120: Literal[0, 1] = Field(..., alias="FBS over 120")
    EKG_results: Literal[0, 1, 2] = Field(..., alias="EKG results")
    Exercise_angina: Literal[0, 1] = Field(..., alias="Exercise angina")
    Thallium: Literal[3, 6, 7]
    Slope_of_ST: Literal[1, 2, 3] = Field(..., alias="Slope of ST")

@app.on_event("startup")
def load_champion_model():
    global model
    try:
        model = mlflow.sklearn.load_model("models:/Heart-Disease-Prediction@champion")
        print("Champion model loaded successfully")
    except Exception as e:
        print(f"Model failed to load: {e}")

def promote_to_champion(run_id: str):
    MODEL_NAME = "Heart-Disease-Prediction"
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)

    load_champion_model()

    return mv.version

@app.post("/manage/promote")
def promote_endpoint(run_id: str):
    try:
        version = promote_to_champion(run_id=run_id)
        return {
            "status": "success", 
            "message": "Model aliased successfully",
            "new_champion_version": str(version)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict")
def predict(data: PatientData):
    data_dict = data.model_dump(by_alias=True)
    input_df = pd.DataFrame([data_dict])

    prediction = model.predict(input_df)[0]
    result = "Presence" if prediction == 1 else "Absence"

    return {"prediction": result}