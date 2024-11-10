import streamlit as st
import pandas as pd
from typing import Dict, List


def display_pipeline(raw_results: Dict[str, Dict[str, List[float]]]) -> None:
    """Method for displaying the pipelines

    Description:
        The reason this is in this file is because it is ran in both
        page 2 AND page 3, and importing a method from a file with an emoji
        made it very difficult.
    Args:
        raw_results (Dict[str, Dict[str, List[float]]]):
        The data of the pipeline, which has to be displayed
    """
    train_metrics_df = pd.DataFrame(raw_results["train_metrics"])
    test_metrics_df = pd.DataFrame(raw_results["test_metrics"])
    test_predictions_df = pd.DataFrame(raw_results["test_predictions"])
    train_predictions_df = pd.DataFrame(raw_results["train_predictions"])

    st.write("## Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Training Metrics")
        st.dataframe(train_metrics_df)

    with col2:
        st.write("### Testing Metrics")
        st.dataframe(test_metrics_df)

    col3, col4 = st.columns(2)

    with col3:
        st.write("### Training Predictions")
        st.dataframe(train_predictions_df)

    with col4:
        st.write("### Testing Predictions")
        st.dataframe(test_predictions_df)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    st.sidebar.success("Select a page above.")
    st.markdown(open("README.md", encoding="utf-8").read())
