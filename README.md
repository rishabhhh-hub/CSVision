# CSV Insight Studio

Upload a CSV file and instantly get:
- Automated EDA (shape, dtypes, missing values, duplicates, summary stats)
- Visual insights (correlation heatmap, histograms, boxplots, top categories) via Matplotlib + Seaborn
- One-click ML model training (regression or classification) via scikit-learn, with evaluation metrics

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## How it works

1. **Upload** a `.csv` file on the home page (drag-and-drop or click to browse).
2. The app parses it with **Pandas** and generates:
   - Row/column counts, dtypes, missing-value report, duplicate count
   - Summary statistics table for numeric columns
   - Correlation heatmap, distribution histograms, boxplots, and top-category bar charts
     (rendered with **Matplotlib**/**Seaborn** and embedded directly as images — no files left on disk)
3. On the results page, pick a **target column** and a **task type**
   (Regression or Classification) and click **Train Models**.
4. The app preprocesses the data (imputes missing values, one-hot encodes
   categoricals, scales numeric features), trains two models with
   **scikit-learn**, and shows evaluation metrics:
   - Regression: R², MAE, RMSE + an Actual-vs-Predicted scatter plot
   - Classification: Accuracy, Precision, Recall, F1 + a confusion matrix heatmap

## Project structure

```
csv-insight-studio/
├── app.py                 # Flask app: routes, EDA, ML logic
├── requirements.txt
├── templates/
│   ├── index.html          # Upload page
│   └── results.html        # EDA + ML results page
├── static/
│   └── css/style.css       # Dark-themed UI styling
└── uploads/                # Uploaded CSVs are stored here temporarily (per session)
```

## Notes & next steps

- This uses Flask's built-in dev server — fine for local use, but for
  production deployment use a WSGI server (e.g. `gunicorn app:app`) behind
  a reverse proxy, and set a real `SECRET_KEY` environment variable.
- Uploaded files are session-scoped and deleted when you click
  "Upload another file". For a multi-user production deployment, you'd
  want to add file expiry/cleanup and size-based validation.
- Easy extensions: add more model types (SVM, XGBoost), let users pick
  which models to compare, add downloadable PDF/HTML reports, or add
  a "drop columns" step before training.
