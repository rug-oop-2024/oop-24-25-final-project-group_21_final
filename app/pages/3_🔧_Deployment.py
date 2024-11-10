import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
import os
import pandas as pd
import io
from app.Welcome import display_pipeline

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")
st.title("Pipeline Management")

if pipelines:
    pipeline_options = [
        f"{pipeline.name}_v{pipeline.version}" for pipeline in pipelines
    ]
    pipeline_with_version = st.selectbox(
        "Select an existing pipeline", pipeline_options, index=0
    )
    pipeline_name, pipeline_version = pipeline_with_version.split("_v")

    selected_pipeline = pipelines[
        [f"{pipeline.name}_v{pipeline.version}"
         for pipeline in pipelines].index(
            pipeline_with_version
        )
    ]

    save_folder = os.path.join(os.getcwd(), "assets", "saved_pipelines")
    artifact_load_path = os.path.join(
        save_folder, f"{pipeline_name}_v{pipeline_version}.pkl"
    )

    if st.button("Load Saved Pipeline") and st.session_state.pipeline is None:
        st.session_state.pipeline = Pipeline.load(artifact_load_path)
        st.success("Pipeline loaded successfully!")

else:
    st.write("No pipelines found.")

if st.session_state.pipeline is not None:
    loaded_pipeline = st.session_state.pipeline
    input_features = [f.name for f in loaded_pipeline._input_features]
    target_feature = loaded_pipeline._target_feature

    if st.session_state.uploaded_file is None:
        uploaded_file = st.file_uploader("Upload your CSV data", type="csv")
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.success("File uploaded successfully!")

    if st.session_state.uploaded_file is not None:
        uploaded_file = st.session_state.uploaded_file
        file_content = uploaded_file.getvalue()
        input_data = pd.read_csv(io.StringIO(file_content.decode("utf-8")))
        input_feature_names = [f.name for f in loaded_pipeline._input_features]
        input_data = input_data[input_feature_names]

        st.write("File is uploaded!")

        if st.button("Execute Pipeline on this data"):
            results = loaded_pipeline.execute()
            display_pipeline(results)

    else:
        st.write("Please upload a CSV file to continue.")


if st.session_state.pipeline is not None:
    if st.button("Delete Pipeline"):
        automl.registry.delete(selected_pipeline.id)
        st.session_state.pipeline = None
        st.session_state.uploaded_file = None
        st.rerun()
