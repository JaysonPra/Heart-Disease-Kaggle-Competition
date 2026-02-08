from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from typing import List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from src.components.data_preprocessing import DataPreprocessor
from src.components.feature_engineering import FeatureEngineer

def _scaler_mapper(scale_feature: dict, ignore_features: List):
    mapper = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler()
    }

    transformer = []
    for scaler, feature in scale_feature.items():
        active_features = [col for col in feature if col not in ignore_features]
        if active_features:
            scaler_object = mapper.get(scaler.lower())
            if scaler_object:
                transformer.append((scaler, scaler_object, active_features))

    return transformer

def model_pipeline(
        scale_feature: dict, 
        ignore_features:List[str], 
        numerical_features:List[str],
        categorical_features:List[str]
    ) -> Pipeline:
    """Builds the pipeline for training

    Args:
        scale_feature (dict): Map of scaler and features to be scaled.
            Example: {'standard': ['Age', 'BP'], 'minmax': ['Cholesterol]}
        ignore_features (List): Features to be ignored for the pipeline
        numerical_features (List): Numerical Features
        categorical_features (List): Categorical Features

    Returns:
        Pipeline: Pipeline Object for training
    """
    active_categorical = [f for f in categorical_features if f not in ignore_features]

    transformer = _scaler_mapper(scale_feature=scale_feature, ignore_features=ignore_features)

    if active_categorical:
        transformer.append(('categorical', OneHotEncoder(sparse_output=True), active_categorical))
    
    if ignore_features:
        transformer.append(('drop_features', 'drop', ignore_features))

    column_scaler = ColumnTransformer(
        transformers=transformer, 
        remainder="drop"
    )

    pipeline = Pipeline([
        ('preprocessor', DataPreprocessor(categorical_features=categorical_features, numerical_features=numerical_features)),
        ('feature_engineering', FeatureEngineer()),
        ('scaler', column_scaler),
        ('classifier', LogisticRegression()) # Placeholder
    ])

    return pipeline