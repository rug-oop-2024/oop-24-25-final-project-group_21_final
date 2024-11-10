"""
This file contains all model classes from the model folder
"""

from autoop.core.ml.model.model import Model


REGRESSION_MODELS = ["TreeRegression", "KNN_Regression", "SupVecRegression"]
CLASSIFICATION_MODELS = ["TreeClassifier", "KNN_Classifier", "RidgeClass"]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        return REGRESSION_MODELS[model_name]()
    elif model_name in CLASSIFICATION_MODELS:
        return REGRESSION_MODELS[model_name]()
    else:
        raise ValueError("Model not recognized")
