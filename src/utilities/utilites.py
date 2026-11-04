from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from pandas import DataFrame
from typing import List
import joblib
import os



def Create_preprocessor(cfg) :
    """ ## Inputs
    `X` ,`y` Prepare them For Training 
    ## outputs 
    X_train """
    num_pipeline = make_pipeline( StandardScaler() )
    category_pipeline = make_pipeline( OneHotEncoder(drop="if_binary", sparse_output=False,handle_unknown="ignore") )

    category_columns = cfg['data']['category_cols']
    num_columns = cfg['data']['num_cols']

    pipes = ColumnTransformer(transformers=[
        ("num pipe" , num_pipeline , num_columns) ,
        ("category pipe" , category_pipeline , category_columns)
    ])
    processor = Pipeline([
        ( "preprocessor ", pipes) , 
    ]).set_output(transform="pandas")

    return processor

class Preprocess(BaseEstimator , TransformerMixin):
    def __init__( self , cfg ):
        self.Category = cfg['data']["category_cols"]
        self.cfg = cfg

        pipeline_path = os.path.join(
                self.cfg["data"]["pipeline_path"],
                self.cfg["data"]["pipeline_name"]
                        )

        if os.path.exists( pipeline_path ):
            
            self.pipe = joblib.load(pipeline_path)
        else : 
            os.makedirs(cfg['data']["pipeline_path"] , exist_ok=True)
            self.pipe = Create_preprocessor(cfg)


    def fit(self , X:DataFrame , y:DataFrame ):
        
        self.pipe.fit(X,y)

        pipeline_path = os.path.join(
        self.cfg["data"]["pipeline_path"],
        self.cfg["data"]["pipeline_name"]
                )
        
        joblib.dump( self.pipe , pipeline_path )
        return self

    def transform(self , X:DataFrame):
        return self.pipe.transform(X)

            



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