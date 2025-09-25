"""
Compare FeatureService outputs vs CSV for the first row.
Print both values for all REQUIRED_FEATURES without strict CSV overrides.
"""

from datetime import datetime
import pandas as pd
import numpy as np

from feature_service import FeatureService, REQUIRED_FEATURES


def coerce_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(pd.to_numeric(x, errors='coerce'))
        except Exception:
            return np.nan


def main():
    csv_path = 'data/Corrected_Input_Data.csv'
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    lat = float(row['Latitude'])
    lon = float(row['Longitude'])
    event_date = pd.to_datetime(row['event_date']).to_pydatetime()

    svc = FeatureService()
    feats = svc.compute_features(lat, lon, event_date, strict_from_csv=False)

    print(f"Row 0 context: lat={lat}, lon={lon}, date={event_date.date()}")
    print("\nFeature, CSV, FeatureService")
    for k in REQUIRED_FEATURES:
        csv_val = row.get(k)
        csv_val = coerce_float(csv_val)
        svc_val = feats.get(k, np.nan)
        try:
            svc_val = float(svc_val)
        except Exception:
            svc_val = np.nan
        print(f"{k}, {csv_val}, {svc_val}")


if __name__ == '__main__':
    main()



