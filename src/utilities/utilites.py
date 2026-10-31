from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from pandas import DataFrame
from typing import List




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