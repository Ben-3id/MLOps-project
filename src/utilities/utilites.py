from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from pandas import DataFrame
from typing import List
import joblib
import os
import glob
from pathlib import Path
import subprocess
import time
import webbrowser


def run_ui(src_path):

    process = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "ui_used_ai.py"],
        cwd=src_path,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(3)  
    webbrowser.open("http://localhost:8501")

    return process



# function to get datasets however their name
def get_datasets(cfg) -> List:
    root_path = Path(__file__).parent.parent
    datapath = cfg['data']['raw_path']
    return glob.glob(root_path._str +"/"+ datapath + "*.csv")



# function to classifiy a Features to Categorical and numerical
def get_lists(df):
    """
    return Num Columns , Categorical Columns
    """
    num_cols = df.select_dtypes(include="number").columns.to_list()

    category_cols = df.select_dtypes(include=["object", "category"]).columns.to_list()

    return num_cols, category_cols




# function to filter featrues with corr
def Fillter(X:DataFrame , y:DataFrame , threshold) -> DataFrame :
    """
    ### Feature selector 
    select by correlation `corrwith()` <br>
    Takes X , y  return df
    """

    columns = X.columns.to_list()

    for col in columns:
        value = abs(X[[col]].corrwith(y).to_list()[0])
        if value < threshold:
            X.drop(columns=[col] , inplace=True)

    return X




# Filter Transformer to have ability make it like step in Pipeline
class filter(BaseEstimator , TransformerMixin ,auto_wrap_output_keys=("transform",)):

    def __init__(self , threshold:float = 0.1):
        self.threshold = threshold

    def fit(self , example , y):
        self.selected_cols_ = []
        for col in example.columns:
            if example[[col]].corrwith(y).abs().to_list()[0] > self.threshold:
                self.selected_cols_.append(col)
        return self

    def transform(self , example):
        return example[self.selected_cols_]




class TargetPipeline(BaseEstimator , TransformerMixin):

    def __init__(self , pipeline ):

        self.pipeline = pipeline

        self.encoder = OneHotEncoder(
            drop= "if_binary" ,
            sparse_output= False ,
            handle_unknown= "ignore"
        ).set_output(transform="pandas")

    def fit(self , X , y):

        self.pipeline.fit(X , y)
        self.encoder.fit(y)

        return self

    def transform(self , X , y = None):

        newX = self.pipeline.transform(X)

        if y is not None:
            newy = self.encoder.transform(y)
            return newX , newy
        return newX




def Create_preprocessor( example ) :
    """ ## Inputs
    `X` ,`y` Prepare them For Training 
    ## outputs 
    X_train """

    num_pipeline = make_pipeline( StandardScaler() )
    category_pipeline = make_pipeline( OneHotEncoder(drop="if_binary", sparse_output=False,handle_unknown="ignore") )
    num_columns , category_columns = get_lists(example)


    column_pipe = ColumnTransformer(
    [
        ("Category" , category_pipeline , category_columns )
    ] , remainder="passthrough"
)

    pipe = Pipeline([
        ( "preprocessor ", column_pipe) , 
    ]).set_output(transform="pandas")

    processor = TargetPipeline(pipe)

    return processor




class Preprocess(BaseEstimator , TransformerMixin):
    def __init__( self , cfg , example=None):
        self.example = example
        self.cfg = cfg

        self.pipeline_path = os.path.join(
                self.cfg["data"]["pipeline_path"],
                self.cfg["data"]["pipeline_name"]
                        )

        if os.path.exists( self.pipeline_path ):
            
            self.pipe = joblib.load(self.pipeline_path)
        else : 
            os.makedirs(self.cfg['data']["pipeline_path"] , exist_ok=True)
            self.pipe = Create_preprocessor(self.example)


    def fit(self , X:DataFrame , y:DataFrame ):

        self.pipe.fit(X,y)
        
        joblib.dump( self.pipe , self.pipeline_path )
        return self

    def transform(self , X:DataFrame , y:DataFrame = None):
        return self.pipe.transform(X , y)

