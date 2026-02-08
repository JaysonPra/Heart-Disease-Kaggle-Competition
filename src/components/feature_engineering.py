from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["RPP"] = X_copy["BP"] * X_copy["Max HR"] # Failure
        X_copy["Age Adjusted Max HR"] = X_copy["Max HR"] / (220 - X_copy["Age"]) # Failure
        X_copy["Is_Seniors_Risk"] = (X_copy["Age"] > 60).astype(int) # Failure
        X_copy["Lipid_Age_Ratio"] = X_copy["Cholesterol"] / X_copy["Age"] # Failure

        return X_copy