from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List, Optional


class ArtifactRegistry:
    """
    A registry for managing artifacts
    """

    def __init__(self, database: Database, storage: Storage) -> None:
        """Initialize the ArtifactRegistry class

        Args:
            database (Database): The database used for storing data
            storage (Storage):
                The storage used for storing and loading Artifacts
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact by saving it into the storage & database

        Args:
            artifact (Artifact): Artifact which gets registered
        """
        self._storage.save(artifact.data, artifact.asset_path)

        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists the artifacts

        Args:
            type (str, optional): The type of artifacts. Defaults to None.

        Returns:
            List[Artifact]: A list of Artifacts of the specific type
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Gets the artifact by the artifact_id

        Args:
            artifact_id (str): The ID of the artifact which is to be retreived

        Returns:
            Artifact: The artifact object found using the ID
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an Artifact by the artifact_id

        Args:
            artifact_id (str): The ID of the artifact
        """

        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Manages the system which is able to store data

    Attributes:
        _instance (AutoMLSystem): The instance of an AutoMLSystem.
        _storage (LocalStorage): The local storage for managing data.
        _database (Database): The database for managing data.
        _registry (ArtifactRegistry): The artifact registry instance for
                                     managing all data.
    """
    _instance: Optional['AutoMLSystem'] = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes the AutoMLSystem

        Args:
            _storage (LocalStorage): The local storage for managing data.
            _database (Database): The database for managing data.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Returns an instance of AutoMLSystem and creates one

        Returns:
            AutoMLSystem: The instance of the AutoML class
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage(".\\assets\\objects"),
                Database(LocalStorage(".\\assets\\dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """Returns the registry attribute

        Returns:
            ArtifactRegistry: Manager for artifacts
        """
        return self._registry
