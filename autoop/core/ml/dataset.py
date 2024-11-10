from autoop.core.ml.artifact import Artifact
from abc import ABC
import pandas as pd
import io


class Dataset(Artifact, ABC):
    """

    Args:
        Artifact: Inherit methods such as read, save and
        the constructor
        ABC: Allows for abstract base class
    """

    def __init__(self, *args, **kwargs):
        """Initializes the dataset class using the constructor of the Artifact.
        Args:
            *args: positional arguments
            **kwargs: key arguments
        """

        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ):
        """Creates an instance of Dataset from a panda dataframe

        Args:
            data (pd.DataFrame): Dataset in pandas Dataframe format
            name (str): Name of the dataset
            asset_path (str): The path where the dataset is stored
            version (str, optional): Version of the dataset.
                    Defaults to "1.0.0".

        Returns:
            Dataset: an instance of Dataset
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def from_artifact(artifact: Artifact):
        """
        Creates and instance of a Dataset from an Artifact

        Args:
            artifact (Artifact): an instance of Artifact, to be converted

        Returns:
            Dataset: an instance of Dataset
        """

        return Dataset(
            name=artifact.name,
            version=artifact.version,
            asset_path=artifact.asset_path,
            metadata=artifact.metadata,
            tags=artifact.tags,
            data=artifact.data,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset and returns it as a pandas Dataframe

        Returns:
            pd.DataFrame: The converted dataset
        """

        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Saves the pandas dataframe into a CSV format

        Args:
            data (pd.DataFrame): The dataframe to save

        Returns:
            bytes: The CSV data
        """

        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
