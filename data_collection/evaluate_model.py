"""
Evaluate model on Corrected_Input_Data.csv

Outputs:
- Per-class precision/recall/F1 and weighted F1
- Confusion matrix counts
- Average feature values for TP/FP/TN/FN groups
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def main() -> None:
    root = Path(__file__).parent.parent
    csv_path = root / 'data' / 'Corrected_Input_Data.csv'
    model_path = root / 'models' / 'best_model_RandomForest.joblib'

    sys.path.append(str(root / 'data_collection'))
    from feature_service import REQUIRED_FEATURES  # type: ignore

    df = pd.read_csv(csv_path)
    # Drop rows with any missing required features or missing label
    needed_cols = list(REQUIRED_FEATURES) + ['Stability']
    df = df.dropna(subset=[c for c in needed_cols if c in df.columns])
    # Build X strictly from REQUIRED_FEATURES
    X = df[REQUIRED_FEATURES]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
    else:
        # fall back to decision function scaled to [0,1] if needed
        y_score = model.predict(X)
        proba = np.asarray(y_score, dtype=float)

    y_true_raw = df['Stability'] if 'Stability' in df.columns else None
    if y_true_raw is None:
        raise KeyError("'Stability' column not found in CSV")
    # Convert to integers (0/1)
    y_true = (pd.to_numeric(y_true_raw, errors='coerce').fillna(0).astype(int)).values

    y_pred = (proba >= 0.5).astype(int)

    # Metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Confusion groups
    tp_idx = (y_true == 1) & (y_pred == 1)
    tn_idx = (y_true == 0) & (y_pred == 0)
    fp_idx = (y_true == 0) & (y_pred == 1)
    fn_idx = (y_true == 1) & (y_pred == 0)

    def avg_features(mask: np.ndarray) -> pd.Series:
        if mask.sum() == 0:
            return pd.Series({f: np.nan for f in REQUIRED_FEATURES})
        return X.loc[mask, REQUIRED_FEATURES].mean(numeric_only=True)

    avg_tp = avg_features(tp_idx)
    avg_fp = avg_features(fp_idx)
    avg_tn = avg_features(tn_idx)
    avg_fn = avg_features(fn_idx)

    # Pretty print
    print("Classification Report (per-class and weighted):")
    print(json.dumps({
        'precision_0': report.get('0', {}).get('precision', 0.0),
        'recall_0': report.get('0', {}).get('recall', 0.0),
        'f1_0': report.get('0', {}).get('f1-score', 0.0),
        'precision_1': report.get('1', {}).get('precision', 0.0),
        'recall_1': report.get('1', {}).get('recall', 0.0),
        'f1_1': report.get('1', {}).get('f1-score', 0.0),
        'weighted_f1': report.get('weighted avg', {}).get('f1-score', 0.0),
        'support': report.get('accuracy', 0.0)
    }, indent=2))

    print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(cm.tolist())

    def fmt_series(s: pd.Series) -> dict:
        out = {}
        for k, v in s.items():
            try:
                fv = float(v)
                if 'mean_flux' in k:
                    out[k] = f"{fv:.2e}"
                elif 'prcp' in k:
                    out[k] = f"{fv:.2f}"
                elif abs(fv) >= 1000:
                    out[k] = f"{fv:,.0f}"
                elif abs(fv) >= 1:
                    out[k] = f"{fv:.2f}"
                else:
                    out[k] = f"{fv:.6f}"
            except Exception:
                out[k] = str(v)
        return out

    print("\nAverage feature values (TP):")
    print(json.dumps(fmt_series(avg_tp), indent=2))
    print("\nAverage feature values (FP):")
    print(json.dumps(fmt_series(avg_fp), indent=2))
    print("\nAverage feature values (TN):")
    print(json.dumps(fmt_series(avg_tn), indent=2))
    print("\nAverage feature values (FN):")
    print(json.dumps(fmt_series(avg_fn), indent=2))


if __name__ == '__main__':
    main()


