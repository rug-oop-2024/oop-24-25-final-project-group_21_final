from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.

    Description:
        Detects the type of data of features,
        options being numeric and categorical
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """

    df = dataset.read()
    features = [
        Feature(
            name=column,
            type=(
                "numerical"
                if df[column].dtype in ("int64", "float64")
                else "categorical"
            ),
        )
        for column in df.columns
    ]
    return features
