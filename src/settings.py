import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config/config.yaml"

def load_config(Path = CONFIG_PATH):
    with open(Path) as f:
        return yaml.safe_load(f)