from sklearn.svm import SVR
from autoop.core.ml.model import Model
import numpy as np


class SupVecRegression(Model):
    """Object representing the Support Vector Regression Model

    Args:
        Model: Uses the Model baseclas for structure
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the SupVecRegression class

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model = SVR(*args, **kwargs)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Fits the model to the observations

        Args:
            observations (np.ndarray): The input/feature data
            ground_truths (np.ndarray): The true values
        """
        self.model.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values/labels based on the observatioms

        Args:
            observations (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted values
        """
        return self.model.predict(observations)
