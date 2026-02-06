import mlflow
import yaml
import pandas as pd
from config.config import RAW_DATA_DIR
from src.components.model_train import model_trainer
import argparse

def _load_data(data_file):
    return pd.read_csv(data_file)

def _map_target(predictor: pd.Series):
    target_mapping = {
        "Presence": 1,
        "Absence": 0
    }
    
    return predictor.map(target_mapping)

def run_experiment(config_file_location):
    train_data = _load_data(RAW_DATA_DIR / "train.csv")

    X = train_data.copy().drop(['Heart Disease'])
    Y = _map_target(predictor=train_data['Heart Disease'])

    config_file = yaml.safe_load(config_file)

    with mlflow.start_run(
        run_name=config_file['experiment']['name'],
        description=config_file['experiment']['description'],
        nested=True
    ):
        for classifier in config_file['search_grid']:
            grid_search = model_trainer(train_config=config_file)
            
            grid_search.fit(X=X, y=Y)

            mlflow.log_param("Best Estimator", grid_search.best_estimator_)
            
            mlflow.log_metrics(grid_search.cv_results_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Prediction")
    
    parser.add_argument(
        "config",
        type=str,
        help="Write the names of YAML config files"
    )

    parser.parse_args()

    if parser.config:
        run_experiment(parser.config)