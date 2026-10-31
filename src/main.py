# import mlflow
from .settings import load_config
import os
import subprocess
from contextlib import asynccontextmanager

config = load_config()



def StartSpain():

    config = load_config()

    subprocess.Popen(
    ["uv", "run", "mlflow", "ui"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NO_WINDOW,  # Windows
    )
    