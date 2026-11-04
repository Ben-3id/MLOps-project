import mlflow
from .settings import load_config
import os
import joblib
import subprocess
from fastapi import FastAPI
from .routes.api import api_route
from contextlib import asynccontextmanager

config = load_config()



async def StartSpain(app :FastAPI):

    config = load_config()

    subprocess.Popen(
    ["uv", "run", "mlflow", "ui"],
    creationflags=subprocess.CREATE_NEW_CONSOLE,  # Windows
    )
    app.mlflow = mlflow
    app.mlflow.set_tracking_uri("http://127.0.0.1:5000")
    app.mlflow.set_experiment("GPA-Predict")


    app.model = None
    


async def EndSpain(app :FastAPI):
    pass

@asynccontextmanager
async def lifecycle(app :FastAPI):
    await StartSpain(app)

    yield

    await EndSpain(app)


app = FastAPI(lifespan= lifecycle)
app.include_router(api_route)
