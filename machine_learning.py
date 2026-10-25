"""
machine_learning.py
-------------------
Preprocessing pipeline, model training, evaluation, and prediction.
Classification metrics: Accuracy, Precision, Recall, F1 only.
No confusion matrix anywhere in this file.
"""

import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
)


CLASSIFICATION_ALGORITHMS = {
    "logistic_regression":    "Logistic Regression",
    "decision_tree":          "Decision Tree",
    "random_forest":          "Random Forest",
    "knn":                    "K-Nearest Neighbors",
}

REGRESSION_ALGORITHMS = {
    "linear_regression":  "Linear Regression",
    "decision_tree":      "Decision Tree",
    "random_forest":      "Random Forest",
    "knn":                "K-Nearest Neighbors",
}


def get_algorithm_lists():
    return {
        "classification": [
            {"key": k, "label": v} for k, v in CLASSIFICATION_ALGORITHMS.items()
        ],
        "regression": [
            {"key": k, "label": v} for k, v in REGRESSION_ALGORITHMS.items()
        ],
    }


def _build_pipeline(numeric_cols, categorical_cols, algorithm_key, task_type):
    """Build a sklearn Pipeline with preprocessing + model."""
    # Numeric: impute then scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    # Categorical: impute then one-hot encode
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Pick model
    if task_type == "classification":
        model_map = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "decision_tree":       DecisionTreeClassifier(random_state=42),
            "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "knn":                 KNeighborsClassifier(n_neighbors=5),
        }
    else:
        model_map = {
            "linear_regression": LinearRegression(),
            "decision_tree":     DecisionTreeRegressor(random_state=42),
            "random_forest":     RandomForestRegressor(n_estimators=100, random_state=42),
            "knn":               KNeighborsRegressor(n_neighbors=5),
        }

    if algorithm_key not in model_map:
        raise ValueError(f"Unknown algorithm: '{algorithm_key}'.")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_map[algorithm_key]),
    ])


def _validate_inputs(df, target_col, task_type, algorithm_key):
    """Raise ValueError with a user-friendly message if inputs are invalid."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")

    if task_type not in ("classification", "regression"):
        raise ValueError("Task type must be 'classification' or 'regression'.")

    if task_type == "classification" and algorithm_key not in CLASSIFICATION_ALGORITHMS:
        raise ValueError(f"'{algorithm_key}' is not a valid classification algorithm.")
    if task_type == "regression" and algorithm_key not in REGRESSION_ALGORITHMS:
        raise ValueError(f"'{algorithm_key}' is not a valid regression algorithm.")

    y = df[target_col].dropna()
    if len(y) < 20:
        raise ValueError("Need at least 20 non-missing rows in the target column to train.")

    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(
                f"Regression requires a numeric target column. "
                f"'{target_col}' is not numeric."
            )

    if task_type == "classification":
        n_classes = df[target_col].dropna().nunique()
        if n_classes < 2:
            raise ValueError("Classification requires at least 2 unique classes in the target.")
        if n_classes > 50:
            raise ValueError(
                f"The target column has {n_classes} unique values — too many for classification. "
                "Try regression, or choose a column with fewer categories."
            )


def train(df: pd.DataFrame, target_col: str, task_type: str,
          algorithm_key: str, test_size: float = 0.2):
    """
    Train a model and return metrics + charts + feature info for prediction.
    Returns a dict; raises ValueError on bad inputs.
    """
    _validate_inputs(df, target_col, task_type, algorithm_key)

    # --- Prepare X, y ---
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Drop rows where target is missing
    mask = y.notnull()
    X, y = X[mask], y[mask]

    if task_type == "classification":
        y = y.astype(str)

    # Identify feature columns by type
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Cap one-hot explosion
    safe_cat_cols = [
        c for c in categorical_cols if X[c].nunique() <= 50
    ]
    dropped_cat = [c for c in categorical_cols if c not in safe_cat_cols]

    X = X[numeric_cols + safe_cat_cols]

    # Store the original unique values of categorical cols for prediction form
    cat_unique = {
        col: sorted(X[col].dropna().unique().tolist()) for col in safe_cat_cols
    }

    # --- Split ---
    stratify = y if task_type == "classification" and y.nunique() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # --- Build & train pipeline ---
    pipeline = _build_pipeline(numeric_cols, safe_cat_cols, algorithm_key, task_type)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    # --- Metrics ---
    if task_type == "classification":
        avg = "weighted"
        metrics = {
            "Accuracy":  round(float(accuracy_score(y_test, preds)), 4),
            "Precision": round(float(precision_score(y_test, preds, average=avg, zero_division=0)), 4),
            "Recall":    round(float(recall_score(y_test, preds, average=avg, zero_division=0)), 4),
            "F1 Score":  round(float(f1_score(y_test, preds, average=avg, zero_division=0)), 4),
        }
        chart_b64 = None   # no confusion matrix per spec

    else:  # regression
        mse = mean_squared_error(y_test, preds)
        metrics = {
            "R² Score": round(float(r2_score(y_test, preds)), 4),
            "MAE":      round(float(mean_absolute_error(y_test, preds)), 4),
            "MSE":      round(float(mse), 4),
            "RMSE":     round(float(np.sqrt(mse)), 4),
        }
        chart_b64 = _actual_vs_predicted(y_test, preds, algorithm_key)

    return {
        "task_type": task_type,
        "algorithm_key": algorithm_key,
        "algorithm_label": (CLASSIFICATION_ALGORITHMS if task_type == "classification"
                            else REGRESSION_ALGORITHMS)[algorithm_key],
        "metrics": metrics,
        "chart": chart_b64,
        "test_size": test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_cols": numeric_cols + safe_cat_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": safe_cat_cols,
        "cat_unique": cat_unique,
        "dropped_cols": dropped_cat,
        # pipeline is returned so it can be stored in the session store
        "_pipeline": pipeline,
    }


def predict(pipeline, feature_cols: list, numeric_cols: list,
            input_data: dict) -> str | float:
    """
    Run a single-row prediction.
    input_data: {col_name: value, ...}
    """
    row = {}
    for col in feature_cols:
        val = input_data.get(col)
        if col in numeric_cols:
            try:
                row[col] = float(val)
            except (TypeError, ValueError):
                row[col] = np.nan
        else:
            row[col] = str(val) if val is not None else ""

    df_row = pd.DataFrame([row], columns=feature_cols)
    result = pipeline.predict(df_row)
    return result[0]


# ── Internal chart helper ─────────────────────────────────────────────────────

def _actual_vs_predicted(y_test, preds, algorithm_key: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, preds, alpha=0.55, color="#4f6ef7", edgecolors="none", s=25)
    lims = [
        min(float(y_test.min()), float(preds.min())),
        max(float(y_test.max()), float(preds.max())),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    algo_label = REGRESSION_ALGORITHMS.get(algorithm_key, algorithm_key)
    ax.set_title(f"Actual vs Predicted — {algo_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
