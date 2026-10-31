import mlflow.sklearn



EXPERIMENT_URL = "http://127.0.0.1:5000/"
EXPERIMENT_NAME = "GPA-Predict"

mlflow.set_tracking_uri(EXPERIMENT_URL)
mlflow.set_experiment(EXPERIMENT_NAME)


model_uri = "runs:/1deaf569c63545cf8c261b452320b195/model"

model = mlflow.sklearn.load_model(model_uri)

model 