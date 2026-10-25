# CSVision

**CSV Data Insights & Machine Learning Platform**

Upload a CSV file and instantly get EDA, interactive charts, and one-click ML model training — all in your browser, powered by Flask + Scikit-learn.

---

## Features

| Tab | What it does |
|---|---|
| **Overview** | Stat cards (rows, cols, missing, duplicates), scrollable preview table, auto-generated histograms, box plots & correlation heatmap |
| **Data Insights** | AI-generated natural-language bullets, descriptive statistics table, missing-value report, categorical bar breakdowns |
| **Visualizations** | On-demand chart builder — histogram, box plot, scatter, line, count plot, bar chart, correlation heatmap |
| **Machine Learning** | Train classification or regression models (Logistic Regression, Decision Tree, Random Forest, KNN) with a configurable train/test split |
| **Predictions** | Dynamic form built from your trained model's features — enter values and get an instant prediction |

---

## Quick Start

### 1. Clone / unzip

```bash
unzip csvision.zip
cd csvision
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Project Structure

```
csvision/
├── app.py                   # Flask entry point, all API routes
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── data_analysis.py     # EDA: overview, preview, insights
│   ├── visualizations.py    # Chart generation (Matplotlib + Seaborn)
│   └── machine_learning.py  # Sklearn pipelines, training, prediction
├── frontend/
│   ├── index.html           # Single-page app shell
│   ├── style.css            # Light theme with CSS custom properties
│   └── script.js            # All frontend logic (fetch, DOM, forms)
└── uploads/                 # Temporary CSV storage (auto-managed)
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a CSV file |
| `GET` | `/api/overview` | Row/col/missing/duplicate counts |
| `GET` | `/api/preview?n=10` | First N rows (max 50) |
| `GET` | `/api/insights` | Stats, missing, categorical summaries, bullets |
| `POST` | `/api/visualize` | Generate a chart (returns base64 PNG) |
| `POST` | `/api/train` | Train a model, returns metrics |
| `POST` | `/api/predict` | Single-row prediction |
| `GET` | `/api/algorithms` | List available algorithm keys/labels |
| `POST` | `/api/reset` | Clear session data |

---

## Supported Chart Types

Pass `chart_type` in the `/api/visualize` request body:

| `chart_type` | Required params |
|---|---|
| `histogram` | `column` |
| `boxplot` | `column` |
| `scatter` | `x_col`, `y_col` |
| `correlation` | *(none)* |
| `line` | `x_col`, `y_col` |
| `count_plot` | `column` |
| `bar_chart` | `x_col`, `y_col` |
| `auto_histograms` | *(none)* |
| `auto_boxplots` | *(none)* |

---

## Notes

- **File size limit:** 25 MB per CSV upload.
- **Session isolation:** Each browser session gets its own in-memory DataFrame and model; multiple users do not share data.
- **Categorical encoding:** Columns with more than 50 unique values are excluded from ML features to prevent dimensionality explosion.
- **Classification metrics:** Accuracy, Precision, Recall, F1 Score (weighted average).
- **Regression metrics:** R², MAE, MSE, RMSE + Actual vs Predicted scatter chart.

---

## Tech Stack

- **Backend:** Python 3.11+, Flask 3, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Frontend:** Vanilla HTML/CSS/JS, no framework dependencies
- **Charts:** Server-side PNG generation, delivered as base64 strings
