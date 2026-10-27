import mlflow
import pandas as pd
from .models import get_models
from .train import train , eval , get_data
from ..settings import load_config

def run_experiment():
    cfg = load_config()
    model_dict = get_models()
    X_train , X_test , y_train, y_test = get_data(cfg)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("GPA-Predict")

    for model_name , model in model_dict.items():

        with mlflow.start_run():
            mlflow.log_params(model.get_params)

        trained_model = train(X_train , y_train , model )

        metrics = eval(X_test , y_test , trained_model)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=trained_model ,
            name= model_name
        )
        print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    run_experiment()