from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame




def Preprocess(X:DataFrame , y:DataFrame) -> DataFrame :
    """ ## Inputs
    `X` ,`y` Prepare them For Training 
    ## outputs 
    X_train , X_test , y_train , y_test """
    num_pipeline = make_pipeline( StandardScaler() )
    nothing_pipeline = make_pipeline( )
    
    num_columns = X.columns

    processor = Pipeline([
        ( "num pipeline", num_pipeline) , 
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