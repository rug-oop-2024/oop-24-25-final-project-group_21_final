from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model import Model
import numpy as np


class TreeClassifier(Model):
    """
    Object representing the DecisionTreeClassifier model

    Args:
        Model: Uses the Model baseclass for structure
    """
    def __init__(self, **kwargs) -> None:
        """Initializes the TreeRegression model class

        Args:
            **kwargs: Additional key arguments
        """
        super().__init__()
        self.model = DecisionTreeClassifier(**kwargs)

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
