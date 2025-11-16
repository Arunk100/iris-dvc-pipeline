# Week 8 MLOps Assignment — Iris Poisoning Experiments

(Short summary — replace or augment as needed.)

## What I did
- Implemented feature-randomization and label-flip poisoning at 0%, 5%, 10%, 50%.
- Trained a StandardScaler + RandomForest pipeline.
- Logged runs, metrics and artifacts to MLflow (SQLite backend).

## How to run
1. Create a virtualenv and install dependencies:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install scikit-learn pandas numpy mlflow matplotlib seaborn pillow`
2. Set tracking URI and run:
   - `export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"`
   - `python mlops_week8_iris_poisoning.py`
3. Start MLflow UI:
   - `mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --default-artifact-root "$(pwd)/mlruns" --host 0.0.0.0 --port 5000`

## Results
(Insert results / screenshots from MLflow UI here; you can edit this file after pushing.)

