from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN_Classifier(Model):
    """
    Object representing the KNeighborsClassifier model

    Args:
        Model: Uses the Model baseclass for structure
    """
    def __init__(self, k=5, **kwargs) -> None:
        """Initializes the KNeighborsRegressor model

        Args:
            k (int, optional): The amount of neighbors for KNN. Defaults to 5.
            **kwargs: Additional key arguments
        """
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=k, **kwargs)

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray):
        """
        Fits the model to the observations

        Args:
            observations (np.ndarray): The input/feature data
            ground_truths (np.ndarray): The true values
        """
        self.model.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values/labels based on the observations

        Args:
            observations (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted values
        """
        return self.model.predict(observations)
