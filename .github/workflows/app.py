"""
CSV Insight Studio
------------------
Upload a CSV, get automated EDA (stats, missing values, correlations,
distributions) plus optional ML model training (regression/classification)
with evaluation metrics -- all rendered in the browser.
"""

import os
import io
import base64
import uuid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed on the server
import matplotlib.pyplot as plt
import seaborn as sns

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"csv"}
MAX_ROWS_PREVIEW = 10
MAX_NUMERIC_HIST = 12       # cap on number of histograms drawn
MAX_CATEGORICAL_PLOTS = 6   # cap on number of bar charts drawn

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG string for inline HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def get_current_df():
    """Load the DataFrame for the current session from disk, or None."""
    file_id = session.get("file_id")
    if not file_id:
        return None, None
    path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.csv")
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None
    return df, session.get("original_filename", "data.csv")


def build_eda(df):
    """Return a dict of everything the results page needs to display."""
    insights = {}

    insights["shape"] = df.shape
    insights["columns"] = list(df.columns)
    insights["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    insights["missing"] = [
        {"column": col, "missing": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    ]

    # Duplicate rows
    insights["duplicates"] = int(df.duplicated().sum())

    # Summary statistics (numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    insights["numeric_cols"] = list(numeric_df.columns)
    insights["categorical_cols"] = list(categorical_df.columns)

    if not numeric_df.empty:
        desc = numeric_df.describe().T.round(3)
        desc_records = desc.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")
        insights["describe"] = desc_records
    else:
        insights["describe"] = []

    # Preview rows
    insights["preview"] = df.head(MAX_ROWS_PREVIEW).to_dict(orient="records")

    # ---- Charts ----
    charts = {}

    # Correlation heatmap
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(min(1.1 * numeric_df.shape[1] + 2, 12), min(1.1 * numeric_df.shape[1] + 2, 10)))
        corr = numeric_df.corr(numeric_only=True)
        sns.heatmap(corr, annot=corr.shape[0] <= 15, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Correlation Heatmap")
        charts["correlation"] = fig_to_base64(fig)

    # Histograms for numeric columns
    hist_cols = list(numeric_df.columns)[:MAX_NUMERIC_HIST]
    if hist_cols:
        n = len(hist_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(hist_cols):
            sns.histplot(numeric_df[col].dropna(), kde=True, ax=axes[i], color="#4C72B0")
            axes[i].set_title(col, fontsize=10)
        for j in range(len(hist_cols), len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Numeric Distributions", y=1.02)
        fig.tight_layout()
        charts["histograms"] = fig_to_base64(fig)

    # Boxplots for outlier detection
    if hist_cols:
        n = len(hist_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(hist_cols):
            sns.boxplot(x=numeric_df[col].dropna(), ax=axes[i], color="#DD8452")
            axes[i].set_title(col, fontsize=10)
        for j in range(len(hist_cols), len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Outlier Check (Boxplots)", y=1.02)
        fig.tight_layout()
        charts["boxplots"] = fig_to_base64(fig)

    # Bar charts for top categorical columns (low-cardinality only)
    cat_cols_to_plot = [
        c for c in categorical_df.columns
        if categorical_df[c].nunique() <= 20
    ][:MAX_CATEGORICAL_PLOTS]
    if cat_cols_to_plot:
        n = len(cat_cols_to_plot)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(cat_cols_to_plot):
            vc = categorical_df[col].value_counts().head(15)
            sns.barplot(x=vc.values, y=vc.index.astype(str), ax=axes[i], color="#55A868")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("Count")
        for j in range(len(cat_cols_to_plot), len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Top Categories", y=1.02)
        fig.tight_layout()
        charts["categorical"] = fig_to_base64(fig)

    insights["charts"] = charts
    return insights


def prepare_features(df, target_col):
    """Basic preprocessing: impute, one-hot encode, scale numeric features."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Drop rows where target is missing
    mask = y.notnull()
    X, y = X[mask], y[mask]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="mean")
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Cap dimensionality explosion from one-hot encoding
    if X.shape[1] > 200:
        X = X.iloc[:, :200]

    if numeric_cols:
        scaler = StandardScaler()
        cols_present = [c for c in numeric_cols if c in X.columns]
        if cols_present:
            X[cols_present] = scaler.fit_transform(X[cols_present])

    return X, y


def train_and_evaluate(df, target_col, task_type):
    X, y = prepare_features(df, target_col)

    if task_type == "classification":
        y = y.astype(str)

    if len(X) < 10:
        raise ValueError("Not enough rows (need at least 10) to train a model reliably.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {"task_type": task_type, "models": []}

    if task_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results["models"].append({
                "name": name,
                "metrics": {
                    "R2 Score": round(r2_score(y_test, preds), 4),
                    "MAE": round(mean_absolute_error(y_test, preds), 4),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
                }
            })

        # Actual vs Predicted plot for the best (last-trained RF) model
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, preds, alpha=0.6, color="#4C72B0")
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted (Random Forest)")
        results["chart"] = fig_to_base64(fig)

    else:  # classification
        n_classes = y.nunique()
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42),
        }
        avg_method = "binary" if n_classes == 2 else "weighted"
        # For binary metrics, sklearn needs pos_label handling with string classes -> use weighted to be safe
        avg_method = "weighted"

        last_preds, last_labels = None, None
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results["models"].append({
                "name": name,
                "metrics": {
                    "Accuracy": round(accuracy_score(y_test, preds), 4),
                    "Precision": round(precision_score(y_test, preds, average=avg_method, zero_division=0), 4),
                    "Recall": round(recall_score(y_test, preds, average=avg_method, zero_division=0), 4),
                    "F1 Score": round(f1_score(y_test, preds, average=avg_method, zero_division=0), 4),
                }
            })
            last_preds, last_labels = preds, sorted(y.unique())

        cm = confusion_matrix(y_test, last_preds, labels=last_labels)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=last_labels, yticklabels=last_labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Random Forest)")
        results["chart"] = fig_to_base64(fig)

    return results


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Please upload a .csv file.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.csv")
    file.save(save_path)

    # Validate it's actually parseable
    try:
        pd.read_csv(save_path, nrows=5)
    except Exception as e:
        os.remove(save_path)
        flash(f"Could not parse CSV: {e}")
        return redirect(url_for("index"))

    session["file_id"] = file_id
    session["original_filename"] = filename
    return redirect(url_for("results"))


@app.route("/results", methods=["GET"])
def results():
    df, filename = get_current_df()
    if df is None:
        flash("Please upload a CSV file first.")
        return redirect(url_for("index"))

    insights = build_eda(df)
    numeric_cols = insights["numeric_cols"]
    categorical_cols = insights["categorical_cols"]
    ml_target_candidates = numeric_cols + categorical_cols

    return render_template(
        "results.html",
        filename=filename,
        insights=insights,
        ml_target_candidates=ml_target_candidates,
    )


@app.route("/train", methods=["POST"])
def train():
    df, filename = get_current_df()
    if df is None:
        flash("Please upload a CSV file first.")
        return redirect(url_for("index"))

    target_col = request.form.get("target_col")
    task_type = request.form.get("task_type")

    if not target_col or target_col not in df.columns:
        flash("Please choose a valid target column.")
        return redirect(url_for("results"))

    try:
        ml_results = train_and_evaluate(df, target_col, task_type)
        ml_error = None
    except Exception as e:
        ml_results = None
        ml_error = str(e)

    insights = build_eda(df)
    numeric_cols = insights["numeric_cols"]
    categorical_cols = insights["categorical_cols"]
    ml_target_candidates = numeric_cols + categorical_cols

    return render_template(
        "results.html",
        filename=filename,
        insights=insights,
        ml_target_candidates=ml_target_candidates,
        ml_results=ml_results,
        ml_error=ml_error,
        selected_target=target_col,
        selected_task=task_type,
    )


@app.route("/reset", methods=["POST"])
def reset():
    file_id = session.pop("file_id", None)
    session.pop("original_filename", None)
    if file_id:
        path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.csv")
        if os.path.exists(path):
            os.remove(path)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
