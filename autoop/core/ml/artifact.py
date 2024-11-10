import base64
from typing import Optional
from copy import deepcopy


class Artifact:
    """
    Representing Artifact class

    Description:
        Serves as a basemodel for saving and reading datasets and artifacts

    Attributes:
        name (str): Name of the artifact
        asset_path (str): The path to a file to be interacted with
        data (Optional[bytes]): Contents of the file in binary form
        version (str): The version of the artifact
        type (str): The type of the artifact (e.g. pipeline, dataset)
        tags (list[str]): A list of tags
        metadata (dict): Contains additional data about the artifact
    """

    def __init__(
        self,
        name: str,
        asset_path: str,
        data: Optional[bytes],
        version: str,
        type: str,
        tags: list[str] = None,
        metadata: dict = None,
    ):
        """Initializes Artifact object

        Args:
            name (str): Name of the artifact
            asset_path (str): The path to a file to be interacted with
            data (Optional[bytes]): Contents of the file in binary form
            version (str): The version of the artifact
            type (str): The type of the artifact (e.g. pipeline, dataset)
            tags (list[str]): A list of tags
            metadata (dict): Contains additional data about the artifact
        """
        self._name = name
        self._version = version
        self._asset_path = asset_path
        self._type = type
        self._metadata = metadata or {}
        self._tags = tags or []
        self._data = data
        self._id = self.generate_id()

    @property
    def name(self) -> str:
        """Getter for the 'name' attribute."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Setter for the 'name' attribute."""
        self._name = value

    @property
    def version(self) -> str:
        """Getter for the 'version' attribute."""
        return self._version

    @version.setter
    def version(self, value: str):
        """Setter for the 'version' attribute."""
        self._version = value

    @property
    def asset_path(self) -> str:
        """Getter for the 'asset_path' attribute."""
        return self._asset_path

    @asset_path.setter
    def asset_path(self, value: str):
        """Setter for the 'asset_path' attribute."""
        self._asset_path = value

    @property
    def type(self) -> str:
        """Getter for the 'type' attribute."""
        return self._type

    @type.setter
    def type(self, value: str):
        """Setter for the 'type' attribute."""
        self._type = value

    @property
    def tags(self) -> list[str]:
        """Getter for the 'tags' attribute."""
        return deepcopy(self._tags)  # Use deepcopy for mutable lists

    @tags.setter
    def tags(self, value: list[str]):
        """Setter for the 'tags' attribute."""
        self._tags = value

    @property
    def metadata(self) -> dict:
        """Getter for the 'metadata' attribute."""
        return deepcopy(self._metadata)  # Use deepcopy for mutable dicts

    @metadata.setter
    def metadata(self, value: dict):
        """Setter for the 'metadata' attribute."""
        self._metadata = value

    @property
    def data(self) -> Optional[bytes]:
        """Getter for the 'data' attribute."""
        return self._data

    @data.setter
    def data(self, value: Optional[bytes]):
        """Setter for the 'data' attribute."""
        self._data = value

    @property
    def id(self) -> str:
        """Getter for the 'id' attribute."""
        return self._id

    def remove_special_char(self, id: str) -> str:
        """Removes special characters from the generate_id method

        Args:
            id (str): To be cleaned ID

        Returns:
            id (str): Cleaned ID
        """

        id = id.rstrip("=")
        for char in [";", ".", ",", "=", "-", ":"]:
            id = id.replace(char, "_")
        return id

    def generate_id(self):
        """Generates an ID based on the name and version of the Artifact

        Returns:
            encoded_id (str): A unique ID, encoded using base64
        """
        encoded_name = base64.b64encode(self.name.encode()).decode()
        encoded_version = base64.b64encode(self.version.encode()).decode()
        encoded_id = self.remove_special_char(
            f"{encoded_name}_{encoded_version}")
        return encoded_id

    def read(self) -> Optional[bytes]:
        """Reads the data of the artifact

        Returns:
            self.data (Optional[bytes]): the data of the artifact"""
        return self.data

    def save(self, bytes: Optional[bytes]) -> Optional[bytes]:
        """
        Saving the bytes into the file

        Args:
            bytes (bytes): The binary data which to be saved
        Description:
            Opens the file from the path in write & binary mode,
            ensuring consistancy and a safe structure
        """

        self.data = bytes
        return self.read
