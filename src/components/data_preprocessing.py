from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features: List, numerical_features: List):
        pass

    def fit(self, X:pd.DataFrame, y:pd.DataFrame):
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()

        X_copy[categorical_features] = X_copy[categorical_features].astype('category')

        floats_to_convert = ['ST depression']
        ints_to_convert = [col for col in self.numerical_features if col not in floats_to_convert]

        if floats_to_convert in X_copy.columns:
            X_copy[floats_to_convert] = X_copy[floats_to_convert].astype('float32')
        X_copy[ints_to_convert] = X_copy[ints_to_convert].astype('int32')

        return X_copy