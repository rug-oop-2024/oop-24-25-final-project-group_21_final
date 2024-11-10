import streamlit as st
import pandas as pd
import os

from app.core.system import AutoMLSystem
from app.Welcome import display_pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.classification import (
    KNN_Classifier,
    TreeClassifier,
    RidgeClass,
)
from autoop.core.ml.model.regression import (
    TreeRegression,
    KNeighborsRegressor,
    SupVecRegression,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.metric import METRICS, get_metric
from autoop.functional.feature import Feature, detect_feature_types

save_folder = os.path.join(os.getcwd(), "assets", "saved_pipelines")

os.makedirs(save_folder, exist_ok=True)

st.set_page_config(page_title="Modelling", page_icon="📈")

listofmodels_categorical = [TreeClassifier(),
                            KNN_Classifier(),
                            RidgeClass()]

listofmodels_regression = [TreeRegression(),
                           KNeighborsRegressor(),
                           SupVecRegression()]

listofmodels = listofmodels_categorical + listofmodels_regression


def display_pipeline_attributes(pipeline, dataset: Dataset):
    """
    This function displays the attributes
    of the pipeline object in a readable format.
    """
    st.write("### Other Pipeline Attributes")
    st.write(f"Dataset model: {pipeline.model.__class__.__name__}")
    st.write(
        f"Dataset input features: {', '.join(
            f.name for f in pipeline._input_features)}"
    )
    st.write(f"Target features: {pipeline._target_feature.name}")
    st.write(f"Dataset Name: {dataset.name}")
    st.write(f"Split Ratio: {pipeline._split * 100}%")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to"
    "train a model on a dataset."
)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")
if not datasets:
    st.write("PLEASE UPLOAD A DATASET FIRST")
    st.stop()

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None
if "own_data" not in st.session_state:
    st.session_state["own_data"] = False
if "prediction_output" not in st.session_state:
    st.session_state["prediction_output"] = None

if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox(
        "Select an existing dataset", dataset_names, index=0
    )
    selected_dataset = datasets[dataset_names.index(selected_dataset_name)]

    data = pd.read_csv(f"assets/objects/{selected_dataset.asset_path}")
    data_artifact = Dataset.from_artifact(selected_dataset)
    features = detect_feature_types(data_artifact)

    features_dict = {feature.name: feature for feature in features}
    feature_types_dict = {feature.name: feature.type for feature in features}

    selected_ground_truth_name = st.selectbox(
        "What do you want to predict?", list(feature_types_dict.keys())
    )
    ground_type = features_dict.get(selected_ground_truth_name).type
    y = features_dict.get(selected_ground_truth_name)

    selected_features_name = st.multiselect(
        "What features you want to consider?",
        [name for name in features_dict.keys() if (
            name != selected_ground_truth_name)],
    )

if selected_features_name and selected_ground_truth_name:
    target_feature = Feature(name=selected_ground_truth_name, type=ground_type)
    input_feature_list = [
        Feature(name=name, type=feature_types_dict[name])
        for name in selected_features_name
    ]

    model_dict = {model.__class__.__name__: model for model in listofmodels}
    if ground_type == "numerical":
        selected_model_name = st.selectbox(
            "Select a model",
            [model.__class__.__name__ for model in listofmodels_regression],
            index=0,
        )
    else:
        selected_model_name = st.selectbox(
            "Select a model",
            [model.__class__.__name__ for model in listofmodels_categorical],
            index=0,
        )

    selected_model = model_dict[selected_model_name]

    if isinstance(selected_model, (KNN_Classifier, KNeighborsRegressor)):
        k_value = st.slider(
            "Select the number of neighbors (K)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )
        selected_model.set_params(n_neighbors=k_value)

    split_value = st.slider(
        "Select the training data split ratio (0.0 - 1.0)",
        min_value=0.1,
        max_value=0.9,
        value=0.8,
    )
    metrics = [get_metric(name) for name in METRICS]

    if st.button("Run on provided data"):
        st.session_state["pipeline"] = Pipeline(
            dataset=data_artifact,
            model=selected_model,
            input_features=input_feature_list,
            target_feature=target_feature,
            split=split_value,
            metrics=metrics,
        )

        pipeline = st.session_state["pipeline"]

        raw_results = pipeline.execute()
        display_pipeline(raw_results)
        display_pipeline_attributes(pipeline, data_artifact)

    with st.form("save_pipeline_form"):
        st.write("### Save Pipeline")
        pipeline_name = st.text_input("Enter Pipeline Name", "MyPipeline")
        pipeline_version = st.text_input("Enter Pipeline Version", "1.0")
        save_button = st.form_submit_button("Save Pipeline")

    if save_button:
        if st.session_state["pipeline"]:
            st.session_state["pipeline_name"] = pipeline_name
            st.session_state["pipeline_version"] = pipeline_version

            artifact_save_path = os.path.join(
                save_folder, f"{pipeline_name}_v{pipeline_version}.pkl"
            )
            pipeline = st.session_state["pipeline"]
            artifact = pipeline.save(
                pipeline_name, pipeline_version, artifact_save_path
            )

            automl.registry.register(artifact)
            st.success("Succesfully saved your pipeline")
        else:
            st.error(
                "No pipeline has been created."
                "Please run the pipeline first.")

    if st.button("Get own data"):
        st.session_state["own_data"] = True

    if st.session_state["own_data"]:
        st.write("## Predict a New Sample")
        with st.form("prediction_form"):
            user_input = {
                feature: (
                    st.number_input(f"Enter value for {feature}")
                    if feature_types_dict[feature] == "numerical"
                    else st.selectbox(
                        f"Select value for {feature}", data[feature].unique()
                    )
                )
                for feature in selected_features_name
            }
            submit_button = st.form_submit_button("Predict")

        if submit_button:
            X = data[selected_features_name]
            y = data[selected_ground_truth_name]
            selected_model.fit(X, y)

            input_df = pd.DataFrame([user_input])
            prediction = selected_model.predict(input_df)
            st.session_state["prediction_output"] = prediction[0]
            st.write("### Prediction Output")
            st.write(
                "The model predicts: "
                f"{selected_ground_truth_name} = "
                f"{st.session_state['prediction_output']}"
            )
