from pandas import DataFrame ,Series
from sklearn.base import RegressorMixin
from typing import Optional , Dict , Any
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error 
import numpy as np
from ..settings import load_config
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

def get_data(cfg):

    root_path = Path(__file__).parent.parent
    data_path = os.path.join(root_path , cfg['data']['processed_data_path'])

    X_train = pd.read_csv(
        os.path.join(data_path , "X_train_proced.csv")
    )

    X_test = pd.read_csv(
        os.path.join(data_path , "X_test_proced.csv")
    )

    y_train = pd.read_csv(
        os.path.join(data_path , "y_train_proced.csv")
    )

    y_test = pd.read_csv(
        os.path.join(data_path , "y_test_proced.csv")
    )

    return X_train , X_test , y_train, y_test


def train(
        X : DataFrame ,
        y :  DataFrame| Series,
        Model : RegressorMixin ) -> RegressorMixin:

    
    Model.fit(X , y)

    return Model


def train_RS(
        X : DataFrame ,
        y :  DataFrame| Series,
        Model : RegressorMixin ,
        params : Dict) -> RegressorMixin:
    
    search = RandomizedSearchCV(
        estimator=Model,
        param_distributions=params,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    
    
    search.fit(X , y)

    return search.best_estimator_


def eval(
        X : DataFrame ,
        y :  DataFrame| Series,
        Model : RegressorMixin,   
):
    preds = Model.predict(X)

    r2 = r2_score(y , preds)
    mae = mean_absolute_error(y,preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    return {"R2":r2 , "MAE":mae , "RMSE":rmse}