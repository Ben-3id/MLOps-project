import mlflow
from .settings import load_config
import os
from pathlib import Path
import subprocess
from fastapi import FastAPI
from .routes.api import api_route
from contextlib import asynccontextmanager
from .utilities import run_ui , Preprocess
config = load_config()



async def StartSpain(app :FastAPI):

    config = load_config()

    subprocess.Popen(
    ["uv", "run", "mlflow", "ui"],
    creationflags=subprocess.CREATE_NEW_CONSOLE,  # Windows
    )
    src_path = Path(__file__).parent
    print(src_path)
    run_ui(src_path)

    app.processor = Preprocess(config)
    

    app.mlflow = mlflow
    app.mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    app.mlflow.set_experiment(config["mlflow"]["experiment_name"])


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
