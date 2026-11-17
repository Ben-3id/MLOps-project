import mlflow.sklearn
from ..settings import load_config

cfg = load_config()
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


model_uri = "runs:/1deaf569c63545cf8c261b452320b195/model"

model = mlflow.sklearn.load_model(model_uri)

model 