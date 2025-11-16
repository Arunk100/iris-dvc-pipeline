# mlops_week8_iris_poisoning.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import os
import io

def random_feature_poison(X, frac, random_state=None):
    rng = np.random.RandomState(random_state)
    Xp = X.copy()
    n_rows = Xp.shape[0]
    n_poison = int(np.round(frac * n_rows))
    if n_poison == 0:
        return Xp
    idx = rng.choice(n_rows, size=n_poison, replace=False)
    means = Xp.mean(axis=0)
    stds = Xp.std(axis=0) + 1e-6
    for i in idx:
        Xp[i, :] = rng.normal(loc=means, scale=stds)
    return Xp

def label_flip_poison(y, frac, n_classes, random_state=None):
    rng = np.random.RandomState(random_state)
    y_p = y.copy()
    n = y_p.shape[0]
    n_poison = int(np.round(frac * n))
    if n_poison == 0:
        return y_p
    idx = rng.choice(n, size=n_poison, replace=False)
    for i in idx:
        choices = [c for c in range(n_classes) if c != y_p[i]]
        y_p[i] = rng.choice(choices)
    return y_p

def eval_and_log_run(run_name, X_train, X_test, y_train, y_test, poison_desc):
    mlflow.set_experiment("Iris_Poisoning_Experiments")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("poisoning", poison_desc)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision_macro", float(precision))
        mlflow.log_metric("recall_macro", float(recall))
        mlflow.log_metric("f1_macro", float(f1))
        mlflow.sklearn.log_model(pipeline, "model")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_title(f"Confusion Matrix: {run_name}")
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # write image to disk and log as an artifact (BytesIO unsupported by this mlflow version)
        with open("confusion_matrix.png", "wb") as _f:
            _f.write(buf.getbuffer())
        mlflow.log_artifact("confusion_matrix.png")
        plt.close(fig)
        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")
        return {
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": cm,
            "report": report
        }

def run_experiments(random_state=42):
    data = load_iris()
    X0 = data["data"]
    y0 = data["target"]
    n_classes = len(np.unique(y0))
    X_train_clean, X_test, y_train_clean, y_test = train_test_split(
        X0, y0, test_size=0.25, random_state=random_state, stratify=y0
    )
    poison_levels = [0.0, 0.05, 0.10, 0.50]
    results = []
    for p in poison_levels:
        X_train = random_feature_poison(X_train_clean, frac=p, random_state=random_state+int(p*1000))
        y_train = y_train_clean.copy()
        desc = f"feature_noise_{int(p*100)}pct"
        print(f"\n=== RUN: {desc} ===")
        res = eval_and_log_run(run_name=desc, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, poison_desc=desc)
        print("Accuracy:", res["accuracy"])
        print(res["report"])
        results.append((desc, res))
    for p in poison_levels:
        X_train = X_train_clean.copy()
        y_train = label_flip_poison(y_train_clean.copy(), frac=p, n_classes=n_classes, random_state=random_state+1+int(p*1000))
        desc = f"label_flip_{int(p*100)}pct"
        print(f"\n=== RUN: {desc} ===")
        res = eval_and_log_run(run_name=desc, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, poison_desc=desc)
        print("Accuracy:", res["accuracy"])
        print(res["report"])
        results.append((desc, res))
    return results

if __name__ == "__main__":
    results = run_experiments()
    print("\nExperiment runs complete. Check mlruns/ for tracked runs or use `mlflow ui`.")
