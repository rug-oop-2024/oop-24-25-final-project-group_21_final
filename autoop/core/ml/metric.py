from abc import ABC, abstractmethod
import numpy as np


METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
    "recall",
    "precision",
]


def get_metric(name: str):
    """
    Function used for mapping the class/metric to the names

    Args:
        name (str): The name of the metric

    Raises:
        ValueError: If the provided metric name is invalid

    Returns:
        Metric: The instance of the metric class
    """
    match name:
        case "mean_squared_error":
            return MeanSquaredError()
        case "accuracy":
            return Accuracy()
        case "mean_absolute_error":
            return Mean_Absolute_Error()
        case "r_squared":
            return R_Squared()
        case "recall":
            return Recall()
        case "precision":
            return Precision()
        case _:
            raise ValueError("Invalid Metric")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, ground_truths: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Abstract method for calculating the metrics values

        Args:
            ground_truths    (np.ndarray): The true values
            predictions (np.ndarray): The predicted values

        Returns:
            float: Metric result
        """

        pass

    @abstractmethod
    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Abcstractmethod for calculating the metric values

        Args:
            predictions (np.ndarray): The predicted values
            ground_truths (np.ndarray): The true values

        Returns:
            float: The calculated metric value
        """
        pass


class MeanSquaredError(Metric):
    """Computes the Mean Squared Error

    Args:
        Metric: Baseclass for metrics
    """

    def __call__(self, ground_truths: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Calls the evaluate function

        Args:
            ground_truths (np.ndarray): The true values
            predictions (np.ndarray): The predicted values

        Returns:
            float: The calculated Mean squared error
        """
        return self.evaluate(predictions, ground_truths)

    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """Calculates the Mean squared error

        Args:
            predictions (np.ndarray): The predicted values
            ground_truths (np.ndarray): The true values

        Returns:
            float: The calculated mean squared error
        """

        return np.mean((ground_truths - predictions) ** 2)


class Accuracy(Metric):
    """Calculates the Accurary metric

    Args:
        Metric: Basemodel for all metrics
    """

    def __call__(self, ground_truths: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Calls the "evaluate" function to calculate the accuracy.

        Args:
            ground_truths (np.ndarray): The true values.
            predictions (np.ndarray): The predicted values.

        Returns:
            float: The calculated accuracy.
        """

        return self.evaluate(predictions, ground_truths)

    def evaluate(self, predictions, ground_truths):
        """
        Calculates the accuracy.

        Args:
            predictions (np.ndarray): The predicted values.
            ground_truths (np.ndarray): The true values.

        Returns:
            float: The calculated accuracy.
        """
        return np.mean(predictions == ground_truths)


class Mean_Absolute_Error(Metric):
    """Calculates the Mean Absolute Error metric

    Args:
        Metric: Basemodel for all metrics
    """
    def __call__(self, ground_truths: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Calls the "evaluate" function to calculate the mean absolute error.

        Args:
            ground_truths (np.ndarray): The true values.
            predictions (np.ndarray): The predicted values.

        Returns:
            float: The calculated mean absolute error.
        """
        return self.evaluate(predictions, ground_truths)

    def evaluate(self, predictions, ground_truths):
        """
        Calculates the Mean absolute error.

        Args:
            predictions (np.ndarray): The predicted values.
            ground_truths (np.ndarray): The true values.

        Returns:
            float: The calculated mean absolute error.
        """
        return np.mean(np.abs(ground_truths - predictions))


class R_Squared(Metric):
    def __call__(self, ground_truths: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Calls the "evaluate" function to calculate the R-Squared metric.

        Args:
            ground_truths (np.ndarray): The true values.
            predictions (np.ndarray): The predicted values.

        Returns:
            float: The calculated R-Squared.
        """

        return self.evaluate(predictions, ground_truths)

    def evaluate(self, predictions, ground_truths):
        """
        Calculates the R-Squared metric.

        Args:
            predictions (np.ndarray): The predicted values.
            ground_truths (np.ndarray): The true values.

        Returns:
            float: The calculated R-squared metric.
        """

        y_hat = np.mean(ground_truths)
        denominator = np.sum((ground_truths - y_hat) ** 2)
        numerator = np.sum((ground_truths - predictions) ** 2)
        return 1 - (numerator / denominator)


class Precision(Metric):
    """Calculates the Precision Metric

    Args:
        Metric: Basemodel for all metrics
    """
    def __call__(self, ground_truths: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Calls the "evaluate" function to calculate the Precision metric.

        Args:
            ground_truths (np.ndarray): The true values.
            predictions (np.ndarray): The predicted values.

        Returns:
            float: The calculated Precision metric.
        """
        return self.evaluate(predictions, ground_truths)

    def evaluate(self, predictions, ground_truths):
        """
        Calculates the Precision metric.

        Args:
            predictions (np.ndarray): The predicted values.
            ground_truths (np.ndarray): The true values.

        Returns:
            float: The calculated Precision metric.
        """
        classes = np.unique(ground_truths)
        precisions = []

        for cls in classes:
            true_positive = np.sum((predictions == cls)
                                   & (ground_truths == cls))
            predicted_p = np.sum(predictions == cls)

            precision = true_positive / predicted_p if predicted_p > 0 else 0.0
            precisions.append(precision)

        return np.mean(precisions)


class Recall(Metric):
    """Calculates the Recall metric

    Args:
        Metric: Basemodel for all metrics
    """

    def __call__(self, ground_truths: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Calls the "evaluate" function to calculate the Recall metric.

        Args:
            ground_truths (np.ndarray): The true values.
            predictions (np.ndarray): The predicted values.

        Returns:
            float: The calculated Recall metric.
        """
        return self.evaluate(predictions, ground_truths)

    def evaluate(self, predictions, ground_truths):
        """
        Calculates the Recall metric.

        Args:
            predictions (np.ndarray): The predicted values.
            ground_truths (np.ndarray): The true values.

        Returns:
            float: The calculated Recall metric.
        """

        unique_classes = np.unique(ground_truths)
        recalls = []

        for cls in unique_classes:
            true_positive = np.sum((predictions == cls)
                                   & (ground_truths == cls))
            actual_positive = np.sum(ground_truths == cls)

            recall = true_positive / actual_positive if (
                actual_positive > 0) else 0.0
            recalls.append(recall)

        return np.mean(recalls)
