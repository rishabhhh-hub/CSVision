"""
data_analysis.py
----------------
All Pandas-based EDA logic: summary stats, missing values,
correlations, natural-language insight generation.
"""

import numpy as np
import pandas as pd


MAX_ROWS_PREVIEW = 50
MAX_UNIQUE_CATEGORICAL = 50   # skip value-count table for very high cardinality


def build_overview(df: pd.DataFrame) -> dict:
    """High-level numbers shown in stat cards."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_total = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_total": missing_total,
        "duplicates": duplicates,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
    }


def build_preview(df: pd.DataFrame, n: int = 10) -> dict:
    """Return up to n rows and column dtype info."""
    n = min(n, MAX_ROWS_PREVIEW, len(df))
    preview_df = df.head(n)
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "rows": preview_df.where(preview_df.notna(), other=None).to_dict(orient="records"),
        "total_rows": len(df),
    }


def build_insights(df: pd.DataFrame) -> dict:
    """Full EDA: describe, missing, correlations, categorical summaries, insights."""
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    numeric_cols = list(numeric_df.columns)
    categorical_cols = list(categorical_df.columns)

    # --- Descriptive stats ---
    describe = []
    if not numeric_df.empty:
        desc = numeric_df.describe().T.round(4)
        desc["column"] = desc.index
        describe = desc.reset_index(drop=True).to_dict(orient="records")

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = [
        {
            "column": col,
            "missing": int(missing[col]),
            "pct": float(missing_pct[col]),
        }
        for col in df.columns
        if missing[col] > 0
    ]

    # --- Constant / near-constant columns ---
    constant_cols = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]

    # --- Correlation matrix (numeric only) ---
    corr_data = {}
    if len(numeric_cols) >= 2:
        corr = numeric_df.corr(numeric_only=True).round(3)
        corr_data = {
            "columns": numeric_cols,
            "matrix": corr.values.tolist(),
        }

    # --- Categorical summaries ---
    categorical_summary = []
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        top = vc.index[0] if len(vc) > 0 else None
        entry = {
            "column": col,
            "unique": int(df[col].nunique(dropna=False)),
            "most_frequent": str(top) if top is not None else "",
            "most_frequent_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
        }
        # Include value counts only for low-cardinality columns
        if df[col].nunique() <= MAX_UNIQUE_CATEGORICAL:
            entry["value_counts"] = [
                {"value": str(k), "count": int(v)} for k, v in vc.head(20).items()
            ]
        categorical_summary.append(entry)

    # --- Natural-language insights ---
    sentences = _generate_sentences(
        df, numeric_df, categorical_df, missing_info, constant_cols
    )

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "describe": describe,
        "missing": missing_info,
        "constant_cols": constant_cols,
        "correlation": corr_data,
        "categorical_summary": categorical_summary,
        "insights": sentences,
    }


def _generate_sentences(df, numeric_df, categorical_df, missing_info, constant_cols):
    """Generate readable plain-English insight bullets from actual data."""
    sentences = []
    n_rows, n_cols = df.shape

    sentences.append(
        f"The dataset contains {n_rows:,} rows and {n_cols} columns."
    )

    dups = int(df.duplicated().sum())
    if dups:
        sentences.append(f"There are {dups:,} duplicate rows in the dataset.")
    else:
        sentences.append("No duplicate rows were found.")

    if missing_info:
        for m in missing_info[:5]:  # cap to avoid very long lists
            sentences.append(
                f"The '{m['column']}' column has {m['missing']:,} missing "
                f"values ({m['pct']:.1f}% of rows)."
            )
    else:
        sentences.append("No missing values were found in any column.")

    if constant_cols:
        sentences.append(
            f"The following column(s) have only one unique value and may be "
            f"uninformative: {', '.join(constant_cols)}."
        )

    # Highlight highest-correlated numeric pair
    if not numeric_df.empty and len(numeric_df.columns) >= 2:
        corr = numeric_df.corr(numeric_only=True)
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((abs(corr.iloc[i, j]), cols[i], cols[j], corr.iloc[i, j]))
        if pairs:
            pairs.sort(reverse=True)
            top = pairs[0]
            direction = "positively" if top[3] > 0 else "negatively"
            sentences.append(
                f"The strongest correlation is between '{top[1]}' and '{top[2]}' "
                f"({direction} correlated, r = {top[3]:.3f})."
            )

    # Numeric column highlights
    for col in list(numeric_df.columns)[:3]:
        mean_val = numeric_df[col].mean()
        sentences.append(
            f"The '{col}' column has a mean value of {mean_val:,.2f}."
        )

    # Categorical column highlights
    for entry in list(categorical_df.columns)[:2]:
        n_unique = df[entry].nunique()
        sentences.append(
            f"The '{entry}' column has {n_unique} unique categories."
        )

    return sentences
