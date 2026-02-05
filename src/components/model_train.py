from src.pipeline.train_pipeline import model_pipeline
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import List

def _model_mapper(search_grid:List[dict]) -> List[dict]:
    model_map = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier
    }

    new_grid = []
    for entry in search_grid:
        model = entry.copy()
        if 'classifier' in model:
            model['classifier'] = [
                model_map.get(model_name) for model_name in entry['classifier'] 
                if model_name in model_map
            ]
        new_grid.append(model)
    
    return new_grid
        
def model_trainer(train_config: dict, X:pd.DataFrame):
    kfold = StratifiedKFold(
        n_splits=train_config['cv']['n_splits'], 
        shuffle=True,
        random_state=42
    )

    pipeline = model_pipeline(
        scale_feature=train_config['scaler'],
        ignore_features=train_config['feature']['ignore'],
        numerical_features=train_config['feature']['numerical'],
        categorical_features=train_config['feature']['categorical']
    )

    param_grid = _model_mapper(search_grid=train_config['search_grid'])
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=train_config['scoring'],
        n_jobs=-1,
        cv=kfold
    )

    return grid_search