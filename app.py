"""
app.py — CSVision Flask application entry point.
Registers all API routes and serves the single-page frontend.
"""

import os
import uuid

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session, send_from_directory
from werkzeug.utils import secure_filename

from backend.data_analysis import build_overview, build_preview, build_insights
from backend.visualizations import (
    histogram, boxplot, scatter, correlation_heatmap,
    line_chart, count_plot, bar_chart,
    auto_histograms, auto_boxplots,
)
from backend.machine_learning import (
    train, predict, get_algorithm_lists,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "frontend"),
    template_folder=os.path.join(BASE_DIR, "frontend"),
)
app.secret_key = os.environ.get("SECRET_KEY", "csvision-dev-key-change-in-prod")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

# In-process store: {file_id: {"df": DataFrame, "pipeline": ..., "train_meta": ...}}
# Keeps data per session without a database.
_store: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_df():
    """Return the DataFrame for the current session, or None."""
    file_id = session.get("file_id")
    if not file_id or file_id not in _store:
        return None
    return _store[file_id]["df"]


def _err(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


def _json_safe(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (obj != obj):   # NaN
        return None
    return obj


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ---------------------------------------------------------------------------
# API — Upload
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return _err("No file part in the request.")

    f = request.files["file"]
    if not f or f.filename == "":
        return _err("No file selected.")
    if not _allowed(f.filename):
        return _err("Please upload a .csv file.")

    filename = secure_filename(f.filename)
    file_id = str(uuid.uuid4())
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.csv")
    f.save(save_path)

    try:
        df = pd.read_csv(save_path)
    except Exception as exc:
        os.remove(save_path)
        return _err(f"Could not parse CSV: {exc}")

    if df.empty:
        os.remove(save_path)
        return _err("The uploaded CSV file is empty.")

    # Clean up previous session file
    old_id = session.get("file_id")
    if old_id:
        old_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{old_id}.csv")
        if os.path.exists(old_path):
            os.remove(old_path)
        _store.pop(old_id, None)

    session["file_id"] = file_id
    _store[file_id] = {"df": df, "pipeline": None, "train_meta": None}

    return jsonify({
        "message": "File uploaded successfully.",
        "filename": filename,
        "file_id": file_id,
        "rows": len(df),
        "cols": len(df.columns),
    })


# ---------------------------------------------------------------------------
# API — Overview, Preview, Insights
# ---------------------------------------------------------------------------

@app.route("/api/overview", methods=["GET"])
def api_overview():
    df = _get_df()
    if df is None:
        return _err("No dataset loaded. Please upload a CSV first.", 404)
    return jsonify(_json_safe(build_overview(df)))


@app.route("/api/preview", methods=["GET"])
def api_preview():
    df = _get_df()
    if df is None:
        return _err("No dataset loaded. Please upload a CSV first.", 404)
    n = int(request.args.get("n", 10))
    n = max(1, min(n, 50))
    return jsonify(_json_safe(build_preview(df, n)))


@app.route("/api/insights", methods=["GET"])
def api_insights():
    df = _get_df()
    if df is None:
        return _err("No dataset loaded. Please upload a CSV first.", 404)
    return jsonify(_json_safe(build_insights(df)))


# ---------------------------------------------------------------------------
# API — Visualize
# ---------------------------------------------------------------------------

CHART_HANDLERS = {
    "histogram":   lambda df, p: histogram(df, p["column"]),
    "boxplot":     lambda df, p: boxplot(df, p["column"]),
    "scatter":     lambda df, p: scatter(df, p["x_col"], p["y_col"]),
    "correlation": lambda df, p: correlation_heatmap(df),
    "line":        lambda df, p: line_chart(df, p["x_col"], p["y_col"]),
    "count_plot":  lambda df, p: count_plot(df, p["column"]),
    "bar_chart":   lambda df, p: bar_chart(df, p["x_col"], p["y_col"]),
    "auto_histograms": lambda df, p: auto_histograms(df),
    "auto_boxplots":   lambda df, p: auto_boxplots(df),
}


@app.route("/api/visualize", methods=["POST"])
def api_visualize():
    df = _get_df()
    if df is None:
        return _err("No dataset loaded. Please upload a CSV first.", 404)

    body = request.get_json(force=True, silent=True) or {}
    chart_type = body.get("chart_type", "")

    if chart_type not in CHART_HANDLERS:
        return _err(f"Unknown chart type: '{chart_type}'.")

    try:
        img_b64 = CHART_HANDLERS[chart_type](df, body)
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"Chart generation failed: {exc}")

    return jsonify({"image": img_b64})


# ---------------------------------------------------------------------------
# API — Train
# ---------------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def api_train():
    df = _get_df()
    if df is None:
        return _err("No dataset loaded. Please upload a CSV first.", 404)

    body = request.get_json(force=True, silent=True) or {}
    target_col    = body.get("target_col", "")
    task_type     = body.get("task_type", "classification")
    algorithm_key = body.get("algorithm_key", "random_forest")
    test_size     = float(body.get("test_size", 0.2))
    test_size     = max(0.1, min(test_size, 0.5))

    try:
        result = train(df, target_col, task_type, algorithm_key, test_size)
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"Training failed: {exc}")

    # Store pipeline for predictions
    file_id = session["file_id"]
    _store[file_id]["pipeline"] = result.pop("_pipeline")
    _store[file_id]["train_meta"] = {
        "feature_cols":   result["feature_cols"],
        "numeric_cols":   result["numeric_cols"],
        "categorical_cols": result["categorical_cols"],
        "cat_unique":     result["cat_unique"],
        "task_type":      task_type,
    }

    return jsonify(_json_safe(result))


# ---------------------------------------------------------------------------
# API — Predict
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    file_id = session.get("file_id")
    if not file_id or file_id not in _store:
        return _err("No dataset loaded. Please upload a CSV first.", 404)

    pipeline  = _store[file_id].get("pipeline")
    meta      = _store[file_id].get("train_meta")

    if pipeline is None or meta is None:
        return _err("No trained model found. Please train a model first.")

    body = request.get_json(force=True, silent=True) or {}
    input_data = body.get("features", {})

    if not input_data:
        return _err("No feature values provided.")

    try:
        result = predict(
            pipeline,
            meta["feature_cols"],
            meta["numeric_cols"],
            input_data,
        )
        # Convert numpy scalar to Python native
        if isinstance(result, (np.integer,)):
            result = int(result)
        elif isinstance(result, (np.floating,)):
            result = float(result)
        else:
            result = str(result)
    except Exception as exc:
        return _err(f"Prediction failed: {exc}")

    return jsonify({
        "prediction": result,
        "task_type": meta["task_type"],
    })


# ---------------------------------------------------------------------------
# API — Meta (algorithm lists, reset)
# ---------------------------------------------------------------------------

@app.route("/api/algorithms", methods=["GET"])
def api_algorithms():
    return jsonify(get_algorithm_lists())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    file_id = session.pop("file_id", None)
    if file_id:
        path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.csv")
        if os.path.exists(path):
            os.remove(path)
        _store.pop(file_id, None)
    return jsonify({"message": "Session reset."})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
