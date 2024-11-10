from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    def __init__(self, path):
        super().__init__(f"Path not found: {path}")


class Storage(ABC):

    @abstractmethod
    def save(self, data: bytes, path: str):
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str):
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """Implementation of Storage class, used for managing files"""

    def __init__(self, base_path: str = "./assets"):
        """Initializes the LocalStorage

        Args:
            base_path (str): Defaults to "./assets", files are stored here
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str):
        """
        Saving data to a file using a key

        Args:
            data (bytes): Data to save in byte format
            key (str): Path key
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a path using a key

        Args:
            key (str): Path to the data to load

        Returns:
            bytes: Loaded data in bytes format
        """

        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/"):
        """
        Delete the file at the path using the key

        Args:
            key (str): Defaults to "/", the path to delete the data from
        """

        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        Lists all the files from a file path.
        Args:
            prefix (str): Defaults to "/", the directory to list all files from

        Returns:
            List[str]: List of all file paths
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path)
                for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Checks if a path exists

        Args:
            path (str): Path to check of existance

        Raises:
            NotFoundError: If the path is not found

        Returns:
            None: If the file path has been found
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """Converts the path

        Description:
            This method converts the path so it is compatible
            with both Linux, MacOS, Windows. And combines the
            path with the base path of that operating system

        Args:
            path (str): Path to convert

        Returns:
            str: converted path
        """

        return os.path.normpath(os.path.join(self._base_path, path))
