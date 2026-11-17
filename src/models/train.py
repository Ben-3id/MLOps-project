from pandas import DataFrame ,Series
from sklearn.base import  ClassifierMixin
from typing import Optional , Dict , Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
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
        Model : ClassifierMixin ) -> ClassifierMixin:

    
    Model.fit(X , y)

    return Model


def train_RS(
        X : DataFrame ,
        y :  DataFrame| Series,
        Model : ClassifierMixin ,
        params : Dict) -> ClassifierMixin:
    
    search = RandomizedSearchCV(
        estimator=Model,
        param_distributions=params,
        n_iter=20,
        scoring="f1_macro",
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
        Model : ClassifierMixin,   
):
    preds = Model.predict(X)

    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(
            y, preds, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y, preds, average="weighted", zero_division=0
        ),
        "F1": f1_score(
            y, preds, average="weighted", zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(y, preds).tolist(),
        "Classification Report": classification_report(
            y, preds, zero_division=0
        ),
    }