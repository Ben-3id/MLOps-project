# MLOps Project

An experimental machine-learning workflow for the Bank Marketing dataset. The repository combines data preparation with DVC, model experiments and artifact tracking with MLflow, and a FastAPI/Streamlit prediction example.

## Project workflow

The DVC pipeline in [`dvc.yaml`](dvc.yaml) defines four stages:

1. **Splitting** — reads CSV files from `src/data/raw/` and separates the configured target (`y`) from the features.
2. **Feature selection** — filters features using their correlation with the target.
3. **Preprocessing** — makes a train/test split, encodes the target, preprocesses features, and saves the fitted preprocessor.
4. **Training** — trains candidate scikit-learn models and records models and evaluation results in MLflow.

The input dataset currently in the repository is `src/data/raw/bank-full.csv`. Pipeline settings, including the target, split size, random seed, and feature-selection threshold, are in [`src/config/config.yaml`](src/config/config.yaml).

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/data/raw/` | Raw input dataset |
| `src/data/splitted/` | Feature and target data |
| `src/data/selected/` | Selected features and target |
| `src/data/processed/` | Preprocessed train/test data |
| `src/pipelines/` | DVC data-processing stages |
| `src/models/` | Candidate models, training, and experiment logging |
| `src/routes/api.py` | FastAPI routes |
| `src/ui_used_ai.py` | Streamlit UI example |
| `src/notebooks/` | Exploratory analysis and data profiling |

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

Install the project and development dependencies:

```shell
uv sync
```

## Run the data and training pipeline

Start the MLflow tracking server in one terminal:

```shell
uv run mlflow server --host 127.0.0.1 --port 5000
```

Then run the DVC pipeline from the repository root in another terminal:

```shell
uv run dvc repro
```

The pipeline writes intermediate data under `src/data/` and the fitted preprocessor to `src/models/dumps/preprocessor.pkl`. Training logs models and evaluations to the MLflow experiment configured in `src/models/model_pipe.py` (`GPA-Predict`). Open `http://127.0.0.1:5000` to view the MLflow UI.

To run only the data-preparation stages, use:

```shell
uv run dvc repro splitting feature_selecting preprocessing
```

## API and UI

The repository also includes a FastAPI application (`src/main.py`) and a Streamlit UI (`src/ui_used_ai.py`). These are currently examples rather than an end-to-end serving path for the Bank Marketing pipeline: their request fields and labels describe **student performance**, while the training data and model pipeline use the **Bank Marketing** dataset. The model-loading and prediction endpoints therefore need to be aligned with the trained model and its preprocessor before they can be used together.

## Current limitations

- The model candidates in `src/models/models.py` are classifiers, but `src/models/train.py` and the MLflow evaluation configuration use regression metrics/types. Review and reconcile this before interpreting training results.
- `src/config/config.yaml` currently lists regression-model parameter grids, which do not match the classifier names used by the training code.
- The API/UI input schema is unrelated to the Bank Marketing dataset, as noted above.

These mismatches mean the repository should be treated as an in-progress MLOps project; successful execution alone does not establish that its logged metrics or serving predictions are valid.
