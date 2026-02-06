import matplotlib
matplotlib.use('agg')
import mlflow
import yaml
import pandas as pd
import numpy as np
from config.config import RAW_DATA_DIR, EXPERIMENTATION_CONFIG_DIR
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

def _log_feature_importance(best_pipeline):
        final_model = best_pipeline.named_steps['classifier']

        try:
            preprocessor = best_pipeline.named_steps['scaler']
            feature_names = preprocessor.get_feature_names_out()
        except:
            return pd.DataFrame([])

        if hasattr(final_model, 'feature_importances_'):
            importances = final_model.feature_importances_
        elif hasattr(final_model, 'coef_'):
            importances = np.abs(final_model.coef_[0])
        else:
            return pd.DataFrame([])

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        return importance_df    

def run_experiment(config_file_location):
    train_data = _load_data(RAW_DATA_DIR / "train.csv")

    X = train_data.drop(columns=['Heart Disease'])
    Y = _map_target(predictor=train_data['Heart Disease'])

    with open(config_file_location, 'r') as f:
        config_file = yaml.safe_load(f)

    with mlflow.start_run(
        run_name=config_file['experiment']['name'],
        description=config_file['experiment']['description']
    ):
        mlflow.sklearn.autolog(
            log_models=False, 
            log_datasets=False, 
            silent=True, 
            log_input_examples=False, 
            log_post_training_metrics=False
        )

        grid_search = model_trainer(train_config=config_file)
        grid_search.fit(X=X, y=Y)

        best_estimator = grid_search.best_estimator_

        importance_df = _log_feature_importance(best_pipeline=best_estimator)
        if not importance_df.empty:
            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Prediction")
    
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Write the names of YAML config files"
    )

    args = parser.parse_args()
    if args.config:
        config_file = EXPERIMENTATION_CONFIG_DIR / args.config
        run_experiment(config_file)