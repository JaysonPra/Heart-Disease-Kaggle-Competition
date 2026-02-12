# Heart Disease Predictor

A full-stack ML pipeline to predict Heart Disease for the Kaggle Competition playground-series-s6e2

**Core architecture:** Scikit-Learn, Pandas, Pydantic, KaggleAPI, MLFlow, FastAPI

## System Architecture

- **Data Pipeline:** Class-based Preprocessing, Scaling, and Feature Engineering using Pandas
- **Experiment Tracking:**: MLFlow used to track hyperparameters and compare between multiple models used through YAML config files
- **Model Registry:** Using Model Aliases (`@champion`) to decouple model training from the prediction API
- **Deployment:** Using FastAPI to provide APIs for JSON prediction, CSV prediction, and Model Promotion

## Tech Stack

- **Language:** Python 3.13
- **API Framework:** FastAPI
- **ML Framework:** Scikit-Learn, XGBoost
- **Data Exploration & Preprocessing:** Pandas
- **Experimentation:** MLFlow
- **Serialization:** MLFlow

## Lessons Learned

- To use classes for Preprocessing, and Feature Engineering
- To switch between Grid Search and Random Search

## Directory structure:

```
└── Kaggle Heart Disease Competition/
    ├── Dockerfile
    ├── README.md
    ├── config/
    ├── data/               # Raw Data and Processed Test Data
    ├── mlruns/
    ├── src/                # Source Code
    │   ├── components/     # Components for Preprocessing, Feature Engineering, Data Ingestion, etc
    │   ├── main.py         # FastAPI app
    │   └── pipeline/       # Scikit-Learn Pipeline for Model Training and Testing
    ├── notebooks/          # Notebooks for EDA
    ├── pyproject.toml
    ├── requirements.txt
    └── uv.lock
```
