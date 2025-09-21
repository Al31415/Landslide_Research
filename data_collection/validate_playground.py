"""
Validate Playground Logic against Corrected_Input_Data.csv for first two rows.

Compares computed features from FeatureService.compute_features with the CSV
columns for REQUIRED_FEATURES. Bulk Density is allowed to deviate; others
should match within tolerance.
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple

import math
import pandas as pd
import numpy as np

from feature_service import FeatureService, REQUIRED_FEATURES


CSV_FEATURE_MAP: Dict[str, str] = {
    'Slope From USGS Elevation Data': 'Slope From USGS Elevation Data',
    'Slope From SSURGO': 'Slope From SSURGO',
    'max_1_day_prcp': 'max_1_day_prcp',
    'avg_30_day_prcp': 'avg_30_day_prcp',
    'max_3_day_prcp': 'max_3_day_prcp',
    'avg_90_day_prcp_mean_flux': 'avg_90_day_prcp_mean_flux',
    'avg_365_day_prcp_mean_flux': 'avg_365_day_prcp_mean_flux',
    'Bulk Density': 'Bulk Density',
    'avg_60_day_prcp': 'avg_60_day_prcp',
    'avg_90_day_prcp': 'avg_90_day_prcp',
    'max_7_day_prcp': 'max_7_day_prcp',
    'Deepest Soil Horizon Layer': 'Deepest Soil Horizon Layer',
}


def _coerce_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float('nan')
        return float(x)
    except Exception:
        try:
            return float(pd.to_numeric(x, errors='coerce'))
        except Exception:
            return float('nan')


def compare_features(expected: Dict[str, float], actual: Dict[str, float],
                     skip_keys: List[str], atol: float = 1e-6, rtol: float = 1e-2) -> List[Tuple[str, float, float]]:
    mismatches: List[Tuple[str, float, float]] = []
    for key in REQUIRED_FEATURES:
        if key in skip_keys:
            continue
        exp = _coerce_float(expected.get(key))
        act = _coerce_float(actual.get(key))
        if (pd.isna(exp) and pd.isna(act)):
            continue
        if pd.isna(exp) or pd.isna(act):
            mismatches.append((key, exp, act))
            continue
        if not math.isclose(exp, act, rel_tol=rtol, abs_tol=atol):
            mismatches.append((key, exp, act))
    return mismatches


def main() -> None:
    csv_path = 'data/Corrected_Input_Data.csv'
    df = pd.read_csv(csv_path)
    first_two = df.head(2).copy()

    svc = FeatureService()

    overall_ok = True
    reports: List[str] = []

    for idx, row in first_two.iterrows():
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        date_raw = row['event_date']
        event_date = pd.to_datetime(date_raw).to_pydatetime()

        feats = svc.compute_features(lat, lon, event_date, strict_from_csv=True)
        # Build actual dict in REQUIRED_FEATURES order
        actual = {k: feats.get(k, np.nan) for k in REQUIRED_FEATURES}
        expected = {k: _coerce_float(row[CSV_FEATURE_MAP[k]]) for k in REQUIRED_FEATURES if CSV_FEATURE_MAP[k] in row}

        mismatches = compare_features(expected, actual, skip_keys=['Bulk Density'])

        rep = []
        rep.append(f"Row {idx}: lat={lat:.6f}, lon={lon:.6f}, date={event_date.date()}")
        if mismatches:
            overall_ok = False
            rep.append("Mismatches (expected vs actual):")
            for k, exp, act in mismatches:
                rep.append(f"  - {k}: {exp} vs {act}")
        else:
            rep.append("All matched within tolerance (except Bulk Density, allowed to differ).")

        # Always show Bulk Density values
        rep.append(
            f"Bulk Density: expected={_coerce_float(row.get('Bulk Density'))}, actual={_coerce_float(actual.get('Bulk Density'))}"
        )

        reports.append('\n'.join(rep))

    print('\n\n'.join(reports))
    if not overall_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()


