from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal
from abc import ABC, abstractmethod
from pydantic import PrivateAttr


class Model(Artifact, ABC):
    """
    Abstract BaseClass for the regression and classification models

    Attributes:
        _type (Literal["classification", "regression"]): The type of the model.
        _parameters (dict): The parameters for the model.
    """

    _type: Literal["classification", "regression"]
    _parameters: dict = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs) -> None:
        """Initializes the model with optional parameters

        Args:
            **kwargs: Optional key parameters
        """
        super(Artifact, self).__init__(**kwargs)

    @property
    def type(self) -> str:
        """Returns a copy of the models type

        Returns:
           str: The type of the model
        """
        return self._type

    @type.setter
    def type(self, type: str) -> None:
        self._type = type

    @property
    def parameters(self) -> dict:
        """Returns a copy of the models parameters

        Returns:
            dict: A copy of the parameters
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Trains the model on the dataset

        Args:
            x (np.ndarray): Contains the input features
            y (np.ndarray): Contains the labels/ground_truths
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Makes predictions using the model

        Args:
            x (np.ndarray): Contains the input features
            y (np.ndarray): Contains the labels/ground_truths

        Returns:
            np.ndarray: An array with the predictions
        """
        pass
