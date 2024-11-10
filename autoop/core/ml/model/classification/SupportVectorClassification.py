from sklearn.linear_model import RidgeClassifier
from autoop.core.ml.model import Model
import numpy as np


class RidgeClass(Model):
    """Object representing the RidgeClassifier

    Args:
        Model: Uses the Model baseclas for structure
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes the SupVecRegression class

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model = RidgeClassifier(**kwargs)

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
