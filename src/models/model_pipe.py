import mlflow
import pandas as pd
from .models import get_models
from .train import train , eval , get_data , train_RS
from ..settings import load_config

def run_experiment():

    EXPERIMENT_URL = "http://127.0.0.1:5000/"
    EXPERIMENT_NAME = "GPA-Predict"
    BEST_METRIC = "R2"

    cfg = load_config()
    model_dict = get_models(cfg)
    X_train , X_test , y_train, y_test = get_data(cfg)

    train_eval_data = X_train.copy()
    train_eval_data["target"] = y_train.to_numpy().ravel()
    test_eval_data = X_test.copy()
    test_eval_data["target"] = y_test.to_numpy().ravel()

    mlflow.set_tracking_uri(EXPERIMENT_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name , dict in model_dict.items():

        model = dict['model']
        params = dict['params']
        with mlflow.start_run():
            mlflow.log_params(model.get_params())

            trained_model = train_RS(X_train.squeeze() , y_train.squeeze() , model  , params)


            model_info = mlflow.sklearn.log_model(
                    sk_model=trained_model,
                    name=model_name,
                    registered_model_name=model_name,
                    serialization_format="cloudpickle",
                    )


            mlflow.evaluate(
                    model=model_info.model_uri,
                    data=train_eval_data,
                    targets="target",
                    model_type="regressor",
                    evaluator_config={
                        "metric_prefix": "train_"
                    },
                )

            # Evaluate on the test dataset


            mlflow.evaluate(
                    model=model_info.model_uri,
                    data=test_eval_data,
                    targets="target",
                    model_type="regressor",
                    evaluator_config={
                        "metric_prefix": "test_"
                    },
            )
                    


if __name__ == "__main__":
    run_experiment()