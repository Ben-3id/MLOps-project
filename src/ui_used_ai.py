import mlflow
import streamlit as st
import pandas as pd
import joblib
from settings import load_config
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from pandas import DataFrame
from typing import List


cfg =  load_config()


def Preprocess(Category:List , X:DataFrame , y:DataFrame = None ) -> DataFrame :
    """ ## Inputs
    `X` ,`y` Prepare them For Training 
    ## outputs 
    X_train , y """
    num_pipeline = make_pipeline( StandardScaler() )
    category_pipeline = make_pipeline( OneHotEncoder(drop="if_binary", sparse_output=False,handle_unknown="ignore") )

    category_columns = Category
    num_columns = X.columns.difference(category_columns).tolist()

    pipes = ColumnTransformer(transformers=[
        ("num pipe" , num_pipeline , num_columns) ,
        ("category pipe" , category_pipeline , category_columns)
    ])
    processor = Pipeline([
        ( "preprocessor ", pipes) , 
    ]).set_output(transform="pandas")

    return processor.fit_transform(X)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")
    return model


model = load_model()

st.title("🎓 Student Performance Predictor")
st.write(
    "Enter the student's information to predict the Performance Index."
)



with st.form("prediction_form"):
    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        options=["Yes", "No"]
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    sample_papers = st.number_input(
        "Sample Question Papers Practiced",
        min_value=0,
        max_value=100,
        value=4,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict Performance Index",
        use_container_width=True
    )


if submitted:
    input_data = pd.DataFrame(
        [{
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extracurricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": sample_papers,
        }]
    )

    # Match the encoding used during training if needed.
    input_data["Extracurricular Activities"] = (
        input_data["Extracurricular Activities"]
        .map({"Yes": "Yes", "No": "No"})
    )

    try:    
        category_list = cfg['data']['category_cols']
        input_data_trans= Preprocess(X= input_data , Category=category_list)
        prediction = model.predict(input_data_trans)[0]

        st.success("Prediction completed successfully!")
        st.metric(
            label="Predicted Performance Index",
            value=f"{prediction:.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Make sure the model expects these feature names "
            "and the same preprocessing used during training."
        )
