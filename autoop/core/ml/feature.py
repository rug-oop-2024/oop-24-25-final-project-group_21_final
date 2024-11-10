from pydantic import BaseModel, Field


class Feature(BaseModel):
    """
    Class that represents a Feature

    Args:
        BaseModel: Allows for pydantic's Field and PrivateAttr functions
    """

    name: str = Field()
    type: str = Field()

    def __str__(self) -> str:
        """Magic method to make sure when the class is printed,
        it always returns the same structure

        Returns:
            str: A description of the featurename and type
        """

        return f"Feature {self.name} contains {self.type} data"
