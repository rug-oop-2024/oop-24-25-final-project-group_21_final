from typing import List, Dict
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np
import base64


class Pipeline:
    """Pipeline class representing a ML pipeline

    Attributes:
        _dataset (Dataset):
            The dataset for models
        _model (Model):
            The model that gets executed
        _input_features (List[Feature]):
            The input/observation features for the model.
        _target_feature (Feature):
            The target feature/ground_truth for model training.
        _metrics (List[Metric]):
            The metrics used for evaluating the model.
        _split (float):
            The percentage of data used for training, default at 0.8 .
        _artifacts (dict): A dictionary storing artifacts.
    """

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ):
        """Initialize the Pipeline

        Args:
            _dataset (Dataset):
                The dataset for models
            _model (Model):
                The model that gets executed
            _input_features (List[Feature]):
                The input/observation features for the model.
            _target_feature (Feature):
                The target feature/ground_truth for model training.
            _metrics (List[Metric]):
                The metrics used for evaluating the model.
            _split (float):
                The percentage of data used for training, default at 0.8 .
            _artifacts (dict): A dictionary storing artifacts.
        """

        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

    def __str__(self) -> str:
        """String representation of the pipeline class

        Returns:
            str: The string representation of the pipeline
        """
        return f"""
Pipeline(
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def input_features(self) -> List[Feature]:
        """Get the input feature attribute

        Returns:
            List[Feature]: the list of features, which is the attribute
        """
        return self._input_features\


    @input_features.setter
    def input_features(self, input_features: List[Feature]):
        """Set the input feature attribute

        Args:
            input_features (List[Feature]): Input feature to be set
        """

        self._input_features = input_features

    @property
    def target_feature(self) -> Feature:
        """Get the target feature attribute

        Returns:
            Feature: Target feature attribute
        """
        return self._target_feature

    @target_feature.setter
    def target_feature(self, target_feature: Feature):
        """Set the target feature attribute

        Args:
            target_feature (Feature): Target feature to be set
        """
        self._target_feature = target_feature

    def generate_id(self) -> str:
        """Generates a unique ID based on the name and version of the dataset

        Returns:
            str: The unique ID, used for storage
        """
        encoded_name = base64.b64encode(self.name.encode()).decode()
        encoded_version = base64.b64encode(self.version.encode()).decode()
        return self.remove_special_char(f"{encoded_name}_{encoded_version}")

    @property
    def model(self) -> Model:
        """Get the model of the pipeline

        Returns:
            Model: model attribute
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the
        pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )

        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """Registers the artifact by placing it inside the artifact list,
        on the correct position

        Args:
            name (str): The name for the artifact to be registered at
            artifact (Artifact): Artifact object
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocesses the features, depending on its type of data.
        For later execution of the models.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]

        self._register_artifact(target_feature_name, artifact)

        input_results = preprocess_features(
            self._input_features, self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [data for (
            feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """Splits the data around the split values
        for seperating into training and test data"""
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[: int(split * len(
            self._output_vector))]
        self._test_y = self._output_vector[int(split * len(
            self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Converts the vectors into one numpy array
        Args:
            vectors (List[np.array]): The list containing vectors

        Returns:
            np.array: The converted array
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Trains the model using vectors and the model attribute"""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluates the results of the model"""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)

        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> Dict[str, Dict[str, List[float]]]:
        """Performs all necessary methods and finally returns the results

        Returns:
            Dict[str, Dict[str, List[float]]]:
                A dictionary containing the results
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()

        train_x = self._compact_vectors(self._train_X)
        test_x = self._compact_vectors(self._test_X)

        train_predictions = self._model.predict(train_x)

        train_metrics_results = {
            "Metric": [type(metric).__name__ for metric in self._metrics],
            "Value": [
                float(metric.evaluate(train_predictions, self._train_y))
                for metric in self._metrics
            ],
        }

        test_predictions = self._model.predict(test_x)
        test_metrics_results = {
            "Metric": [type(metric).__name__ for metric in self._metrics],
            "Value": [
                float(metric.evaluate(test_predictions, self._test_y))
                for metric in self._metrics
            ],
        }

        return {
            "train_metrics": train_metrics_results,
            "train_predictions": train_predictions,
            "test_metrics": test_metrics_results,
            "test_predictions": test_predictions,
        }

    def save(self, name: str, version: str, save_path: str) -> Artifact:
        """Save the pipeline as an Artifact with specified name and version.

        Args:
            name (str): The name of the artifact
            version (str): The version of the artifact
            save_path (str): The path containing the artifact

        Returns:
            Artifact: The pipeline artifact
        """
        data = pickle.dumps(
            {
                "configuration": {
                    "dataset": self._dataset,
                    "input_features": self._input_features,
                    "target_feature": self._target_feature,
                    "split": self._split,
                    "metrics": self._metrics,
                    "model": self._model,
                }
            }
        )

        artifact = Artifact(
            name=name,
            asset_path=save_path,
            data=data,
            version=version,
            type="pipeline",
        )

        with open(save_path, "wb") as f:
            f.write(data)

        return artifact

    @staticmethod
    def load(file_path: str) -> "Pipeline":
        """Load and reconstructs the pipeline configuration
        from an artifact.

        Args:
            file_path (str): contains the file path to the saved artifacts

        Returns:
            pipeline (Pipeline): The reconstructed pipeline
        """

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        pipeline = Pipeline(
            dataset=data["configuration"]["dataset"],
            model=data["configuration"]["model"],
            input_features=data["configuration"]["input_features"],
            target_feature=data["configuration"]["target_feature"],
            split=data["configuration"]["split"],
            metrics=data["configuration"].get("metrics", []),
        )

        return pipeline
