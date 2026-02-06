from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features: List, numerical_features: List):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def fit(self, X:pd.DataFrame, y:pd.DataFrame):
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()

        existing_cats = [col for col in self.categorical_features if col in X_copy.columns]
        if existing_cats:
            X_copy[existing_cats] = X_copy[existing_cats].astype('category')

        floats_to_convert = ['ST depression']
        ints_to_convert = [col for col in self.numerical_features 
                            if col not in floats_to_convert and col in X_copy.columns]

        if all(col in X_copy.columns for col in floats_to_convert):
            X_copy[floats_to_convert] = X_copy[floats_to_convert].astype('float64')
        X_copy[ints_to_convert] = X_copy[ints_to_convert].astype('int32')

        return X_copy