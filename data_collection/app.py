import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pydeck as pdk
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Stability Predictor", layout="wide")

st.title("US Stability Predictor (CMIP + Meteostat + SSURGO + USGS)")

# Try to pre-load original coordinates
orig_points = None
try:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'data', 'Corrected_Input_Data.csv')
    df_orig = pd.read_csv(csv_path)
    if {'Latitude', 'Longitude'}.issubset(df_orig.columns):
        orig_points = df_orig[['Latitude', 'Longitude']].dropna().head(500)
except Exception:
    pass

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Location and Date")
    default_lat, default_lon = 37.7749, -122.4194
    lat = st.number_input("Latitude", value=float(default_lat), format="%0.6f")
    lon = st.number_input("Longitude", value=float(default_lon), format="%0.6f")
    date = st.date_input("Event Date", value=datetime(2025, 1, 15))
    run = st.button("Compute Prediction")

    st.subheader("Map")
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=5000,
        )
    ]
    if orig_points is not None and not orig_points.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=orig_points.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}),
                get_position='[lon, lat]',
                get_color='[0, 100, 255, 100]',
                get_radius=2000,
            )
        )
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=4 if orig_points is not None else 6,
                pitch=0,
            ),
            layers=layers,
        )
    )

with col2:
    st.subheader("Prediction and Features")
    if run:
        try:
            from feature_service import FeatureService
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(__file__))
            from feature_service import FeatureService
        svc = FeatureService()
        status = st.empty()
        progress_area = st.container()
        progress_log = []

        def on_progress(stage: str, message: str, data: dict):
            progress_log.append({"stage": stage, "message": message, **data})
            status.info(f"{stage}: {message}")

        with st.spinner("Collecting data and computing features..."):
            feats = svc.compute_features(
                lat,
                lon,
                datetime.combine(date, datetime.min.time()),
                progress_callback=on_progress,
            )
            pred = svc.predict(feats)

        st.success("Data collection and prediction complete.")

        with progress_area.expander("Detailed data collection log", expanded=True):
            log_df = pd.DataFrame(progress_log)
            st.dataframe(log_df, use_container_width=True)

        st.metric("Stability Score", f"{pred['stability_score']:.3f}")
        st.write("Stable:" if pred['stable'] else "Unstable:", pred['stable'])

        st.markdown("### Feature Values (with units)")
        units = feats.get('_units', {})
        raw = feats.get('_raw', {})
        display_rows = []
        for k, v in feats.items():
            if k.startswith('_'):
                continue
            display_rows.append({'Feature': k, 'Value': v, 'Units': units.get(k, '')})
        feat_df = pd.DataFrame(display_rows)
        st.dataframe(feat_df, use_container_width=True)

        if 'hzname_values' in raw:
            st.markdown("### SSURGO hzname raw values (sample)")
            st.write(raw['hzname_values'][:20])

        try:
            import shap
            required_features = [
                'Slope From USGS Elevation Data',
                'Slope From SSURGO',
                'max_1_day_prcp',
                'avg_30_day_prcp',
                'max_3_day_prcp',
                'avg_90_day_prcp_mean_flux',
                'avg_365_day_prcp_mean_flux',
                'Bulk Density',
                'avg_60_day_prcp',
                'avg_90_day_prcp',
                'max_7_day_prcp',
                'Deepest Soil Horizon Layer',
            ]

            feature_values = {}
            for feature in required_features:
                value = feats.get(feature, 0.0)
                if pd.isna(value) or value is None:
                    value = 0.0
                feature_values[feature] = float(value)

            X_current = pd.DataFrame([feature_values])

            background_data = []
            for _ in range(10):
                bg_sample = {}
                for feature in required_features:
                    if 'Slope' in feature:
                        bg_sample[feature] = np.random.uniform(0, 45)
                    elif 'prcp' in feature and 'flux' not in feature:
                        bg_sample[feature] = np.random.uniform(0, 50)
                    elif 'flux' in feature:
                        bg_sample[feature] = np.random.uniform(0, 0.001)
                    elif 'Bulk Density' in feature:
                        bg_sample[feature] = np.random.uniform(0.8, 2.0)
                    elif 'Horizon' in feature:
                        bg_sample[feature] = np.random.uniform(1, 6)
                    else:
                        bg_sample[feature] = np.random.uniform(0, 10)
                background_data.append(bg_sample)

            X_background = pd.DataFrame(background_data)

            model = svc.model
            explainer = shap.Explainer(model.predict_proba, X_background)
            shap_values = explainer(X_current)

            st.markdown("### SHAP Feature Contributions")

            if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] == 2:
                shap_vals = shap_values.values[0, :, 1]
                base_val = shap_values.base_values[0, 1]
            else:
                shap_vals = shap_values.values[0] if len(shap_values.values.shape) > 1 else shap_values.values
                base_val = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0

            st.markdown("### Feature Contributions to Prediction")
            contrib_data = []
            for i, feature in enumerate(required_features):
                contrib_data.append({
                    'Feature': feature,
                    'Value': f"{feature_values[feature]:.4f}",
                    'SHAP Contribution': f"{shap_vals[i]:.6f}",
                    'Impact': 'Positive' if shap_vals[i] > 0 else 'Negative' if shap_vals[i] < 0 else 'Neutral',
                })

            contrib_df = pd.DataFrame(contrib_data)
            contrib_df = contrib_df.sort_values('SHAP Contribution', key=lambda x: x.astype(float).abs(), ascending=False)
            st.dataframe(contrib_df, use_container_width=True)

            st.markdown("### Prediction Breakdown")
            total_contribution = float(np.sum(shap_vals))
            final_prediction = base_val + total_contribution

            breakdown_data = [
                {'Component': 'Base Value (Model Average)', 'Value': f"{base_val:.6f}"},
                {'Component': 'Total Feature Contributions', 'Value': f"{total_contribution:.6f}"},
                {'Component': 'Final Prediction', 'Value': f"{final_prediction:.6f}"},
                {'Component': 'Actual Model Output', 'Value': f"{pred['stability_score']:.6f}"},
            ]
            st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)

            if abs(total_contribution) > 1e-6:
                try:
                    shap_explanation = shap.Explanation(
                        values=shap_vals,
                        base_values=base_val,
                        data=np.array(list(feature_values.values())),
                        feature_names=required_features,
                    )

                    fig, _ = plt.subplots(figsize=(10, 8))
                    shap.plots.waterfall(shap_explanation, show=False)
                    st.pyplot(fig)
                except Exception as plot_error:
                    st.info(f"Waterfall plot creation failed: {plot_error}")
            else:
                st.info("SHAP contributions are too small to visualize meaningfully. Check the feature contribution table above.")

        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.info("Showing basic feature importance instead:")

            fallback_data = []
            for feature in required_features:
                value = feats.get(feature, 0.0)
                if pd.isna(value):
                    value = 0.0
                fallback_data.append({
                    'Feature': feature,
                    'Value': f"{float(value):.4f}",
                    'Relative Magnitude': 'High' if abs(float(value)) > 10 else 'Medium' if abs(float(value)) > 1 else 'Low',
                })

            st.dataframe(pd.DataFrame(fallback_data), use_container_width=True)

with st.expander("Validate Playground logic against first 3 CSV rows"):
    if st.button("Run validation (rows 0–2)"):
        try:
            try:
                from feature_service import FeatureService
            except ImportError:
                import sys
                sys.path.append(os.path.dirname(__file__))
                from feature_service import FeatureService
            svc = FeatureService()
            base_dir = os.path.dirname(os.path.dirname(__file__))
            csv_path = os.path.join(base_dir, 'data', 'Corrected_Input_Data.csv')
            df = pd.read_csv(csv_path)
            rows = df.iloc[[0, 1, 2]]
            TOL_SSURGO = 1e-2
            TOL_MET = 1e-2
            TOL_SLOPE = 0.2
            out = []
            for idx, r in rows.iterrows():
                plat = float(r['Latitude'])
                plon = float(r['Longitude'])
                pdate = datetime.strptime(str(r['event_date']), '%Y-%m-%d')
                feats_i = svc.compute_features(plat, plon, pdate)
                res = {
                    'Row': idx,
                    'USGS slope (deg)': feats_i.get('Slope From USGS Elevation Data', np.nan),
                    'USGS slope expected': float(r['Slope From USGS Elevation Data']),
                    'USGS slope OK': abs(float(feats_i.get('Slope From USGS Elevation Data', np.nan)) - float(r['Slope From USGS Elevation Data'])) < TOL_SLOPE,
                    'Bulk Density OK': abs(float(feats_i.get('Bulk Density', np.nan)) - float(r['Bulk Density'])) < TOL_SSURGO,
                    'Slope From SSURGO OK': abs(float(feats_i.get('Slope From SSURGO', np.nan)) - float(r['Slope From SSURGO'])) < TOL_SSURGO,
                    'Deepest Horizon OK': abs(float(feats_i.get('Deepest Soil Horizon Layer', np.nan)) - float(r['Deepest Soil Horizon Layer'])) < TOL_SSURGO,
                    'Max 1 Day OK': abs(float(feats_i.get('max_1_day_prcp', np.nan)) - float(r['max_1_day_prcp'])) < TOL_MET,
                    'Avg 30 Day OK': abs(float(feats_i.get('avg_30_day_prcp', np.nan)) - float(r['avg_30_day_prcp'])) < TOL_MET,
                }
                out.append(res)
            st.dataframe(pd.DataFrame(out), use_container_width=True)
        except Exception as e:
            st.error(f"Validation failed: {e}") 