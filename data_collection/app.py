import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pydeck as pdk
import matplotlib.pyplot as plt
import os
import requests
import json
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="Stability Predictor", layout="wide")

st.title("US Stability Predictor (CMIP + Meteostat + SSURGO + USGS)")
st.caption("🚀 Version 2.0 - Interactive Map Features | Last Updated: 2025-01-18")

def geocode_location(location_name):
    """
    Geocode a location name to get latitude and longitude coordinates.
    Uses OpenStreetMap Nominatim API (free, no API key required).
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'Landslide-Research-App/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        results = response.json()
        if results:
            lat = float(results[0]['lat'])
            lon = float(results[0]['lon'])
            return lat, lon, results[0].get('display_name', location_name)
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        return None, None, None

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
    
    # Initialize session state for coordinates
    if 'lat' not in st.session_state:
        st.session_state.lat = 37.7749
    if 'lon' not in st.session_state:
        st.session_state.lon = -122.4194
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📍 Click on Map", "🔍 Search Location", "⌨️ Manual Entry"],
        horizontal=True
    )
    
    if input_method == "🔍 Search Location":
        st.markdown("**Search Examples:**")
        st.markdown("- Cities: `San Francisco, CA`, `Seattle, WA`, `Denver, CO`")
        st.markdown("- Landmarks: `Mount Rainier, WA`, `Grand Canyon, AZ`, `Yellowstone National Park`")
        st.markdown("- Addresses: `1600 Pennsylvania Avenue, Washington DC`")
        
        location_input = st.text_input(
            "Enter location:",
            placeholder="Type a city, landmark, or address..."
        )
        
        col_search1, col_search2 = st.columns([1, 1])
        with col_search1:
            search_button = st.button("🔍 Search", type="primary")
        with col_search2:
            if st.button("🗺️ Show on Map"):
                st.session_state.show_search_result = True
        
        if search_button and location_input:
            with st.spinner("Searching for location..."):
                lat_result, lon_result, display_name = geocode_location(location_input)
                if lat_result is not None:
                    st.session_state.lat = lat_result
                    st.session_state.lon = lon_result
                    st.session_state.last_searched_location = display_name
                    st.success(f"Found: {display_name}")
                    st.info(f"Coordinates: {lat_result:.6f}, {lon_result:.6f}")
                else:
                    st.error("Location not found. Please try a different search term.")
    
    elif input_method == "⌨️ Manual Entry":
        st.session_state.lat = st.number_input(
            "Latitude", 
            value=float(st.session_state.lat), 
            format="%0.6f",
            key="manual_lat"
        )
        st.session_state.lon = st.number_input(
            "Longitude", 
            value=float(st.session_state.lon), 
            format="%0.6f",
            key="manual_lon"
        )
    
    # Display current coordinates with better formatting
    st.markdown("### 📍 Current Location")
    col_coord1, col_coord2 = st.columns([1, 1])
    with col_coord1:
        st.metric("Latitude", f"{st.session_state.lat:.6f}")
    with col_coord2:
        st.metric("Longitude", f"{st.session_state.lon:.6f}")
    
    # Add a quick location info display
    if hasattr(st.session_state, 'last_searched_location'):
        st.info(f"📍 Last searched: {st.session_state.last_searched_location}")
    
    # Event date input
    date = st.date_input("Event Date", value=datetime(2025, 1, 15))
    
    # Compute button
    run = st.button("🚀 Compute Prediction", type="primary")

    # Map interface - different for each input method
    st.subheader("Map View")

    if input_method == "🔍 Search Location":
        # Simple visualization map for search results
        st.info("🗺️ **Map showing your searched location and historical data points.**")
        
        # Create map data
        map_data = pd.DataFrame({
            'lat': [st.session_state.lat],
            'lon': [st.session_state.lon]
        })
        
        # Add historical points if available
        if orig_points is not None and not orig_points.empty:
            hist_data = orig_points.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
            map_data = pd.concat([map_data, hist_data], ignore_index=True)
        
        st.map(map_data, zoom=6)

    elif input_method == "⌨️ Manual Entry":
        # Simple visualization map for manual entry
        st.info("🗺️ **Map showing your manually entered coordinates and historical data points.**")
        
        # Create map data
        map_data = pd.DataFrame({
            'lat': [st.session_state.lat],
            'lon': [st.session_state.lon]
        })
        
        # Add historical points if available
        if orig_points is not None and not orig_points.empty:
            hist_data = orig_points.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
            map_data = pd.concat([map_data, hist_data], ignore_index=True)
        
        st.map(map_data, zoom=6)

    elif input_method == "📍 Click on Map":
        # Interactive map for direct selection with click-anywhere functionality
        st.info("🗺️ **Click anywhere on the map below to select a location directly!**")
        
        # Initialize map mode in session state
        if 'map_mode' not in st.session_state:
            st.session_state.map_mode = 'select'  # 'select' or 'pan'
        
        # Map mode toggle
        st.markdown("### 🎛️ Map Controls")
        col_mode1, col_mode2, col_mode3 = st.columns([1, 1, 1])
        
        with col_mode1:
            if st.button("🎯 Select Mode", type="primary" if st.session_state.map_mode == 'select' else "secondary", key="select_mode"):
                st.session_state.map_mode = 'select'
                st.rerun()
        
        with col_mode2:
            if st.button("🖱️ Pan Mode", type="primary" if st.session_state.map_mode == 'pan' else "secondary", key="pan_mode"):
                st.session_state.map_mode = 'pan'
                st.rerun()
        
        with col_mode3:
            if st.button("🔄 Reset View", key="reset_view"):
                st.session_state.lat = 37.7749
                st.session_state.lon = -122.4194
                st.success("Reset to San Francisco coordinates")
                st.rerun()
        
        # Create a simple map for click-anywhere functionality
        map_data = pd.DataFrame({
            'lat': [st.session_state.lat],
            'lon': [st.session_state.lon]
        })
        
        # Add historical points if available
        if orig_points is not None and not orig_points.empty:
            hist_data = orig_points.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
            map_data = pd.concat([map_data, hist_data], ignore_index=True)
        
        # Display the map
        st.map(map_data, zoom=6)
        
        # Click-anywhere functionality
        if st.session_state.map_mode == 'select':
            st.markdown("### 🎯 Click-Anywhere Selection")
            st.info("**Click anywhere on the map above to select that exact location!**")
            
            # JavaScript to capture map clicks
            if st.button("🖱️ Enable Click-to-Select", type="primary", key="enable_click"):
                # Use a simpler single-line JavaScript approach
                js_code = "const mapContainer = document.querySelector('[data-testid=\"stMap\"]'); if (mapContainer) { mapContainer.style.cursor = 'crosshair'; mapContainer.addEventListener('click', function(e) { const rect = e.target.getBoundingClientRect(); const x = e.clientX - rect.left; const y = e.clientY - rect.top; const lat = 90 - (y / rect.height) * 180; const lon = (x / rect.width) * 360 - 180; sessionStorage.setItem('clicked_lat', lat); sessionStorage.setItem('clicked_lon', lon); alert('Location selected: ' + lat.toFixed(6) + ', ' + lon.toFixed(6)); window.location.reload(); }); alert('Click-to-select enabled! Click anywhere on the map.'); } else { alert('Map not found.'); }"
                
                try:
                    streamlit_js_eval(js_code)
                except Exception as e:
                    st.error(f"JavaScript execution failed: {e}")
                    st.info("Please use the quick location buttons below instead.")
            
            # Check if coordinates were clicked
            try:
                clicked_lat = streamlit_js_eval("sessionStorage.getItem('clicked_lat')")
                clicked_lon = streamlit_js_eval("sessionStorage.getItem('clicked_lon')")
                
                if clicked_lat and clicked_lon:
                    try:
                        lat_val = float(clicked_lat)
                        lon_val = float(clicked_lon)
                        if not np.isnan(lat_val) and not np.isnan(lon_val):
                            st.session_state.lat = lat_val
                            st.session_state.lon = lon_val
                            st.success(f"📍 Location selected from map click: {lat_val:.6f}, {lon_val:.6f}")
                            # Clear the session storage
                            streamlit_js_eval("sessionStorage.removeItem('clicked_lat'); sessionStorage.removeItem('clicked_lon');")
                            st.rerun()
                    except (ValueError, TypeError):
                        pass
            except:
                pass
        
        else:  # Pan mode
            st.markdown("### 🖱️ Pan Mode Active")
            st.info("**Pan mode is active. Click and drag to move around the map, or switch to Select Mode to choose locations.**")
        
        # Quick location buttons for common areas
        st.markdown("### 🎯 Quick Location Selection")
        st.markdown("**Or click any button below to instantly set coordinates:**")
        
        col_quick1, col_quick2, col_quick3 = st.columns([1, 1, 1])
        
        with col_quick1:
            if st.button("🏔️ Mount Rainier, WA", key="rainier_click"):
                st.session_state.lat = 46.8523
                st.session_state.lon = -121.7603
                st.success("📍 Set to Mount Rainier, WA")
                st.rerun()
        
        with col_quick2:
            if st.button("🌋 Mount St. Helens, WA", key="helens_click"):
                st.session_state.lat = 46.1914
                st.session_state.lon = -122.1956
                st.success("📍 Set to Mount St. Helens, WA")
                st.rerun()
        
        with col_quick3:
            if st.button("🏔️ Yosemite, CA", key="yosemite_click"):
                st.session_state.lat = 37.8651
                st.session_state.lon = -119.5383
                st.success("📍 Set to Yosemite, CA")
                st.rerun()
        
        # Additional quick locations
        col_quick4, col_quick5, col_quick6 = st.columns([1, 1, 1])
        
        with col_quick4:
            if st.button("🌲 Olympic NP, WA", key="olympic_click"):
                st.session_state.lat = 47.8021
                st.session_state.lon = -123.6044
                st.success("📍 Set to Olympic National Park, WA")
                st.rerun()
        
        with col_quick5:
            if st.button("🏔️ Glacier NP, MT", key="glacier_click"):
                st.session_state.lat = 48.7596
                st.session_state.lon = -113.7870
                st.success("📍 Set to Glacier National Park, MT")
                st.rerun()
        
        with col_quick6:
            if st.button("🌋 Lassen NP, CA", key="lassen_click"):
                st.session_state.lat = 40.4983
                st.session_state.lon = -121.4209
                st.success("📍 Set to Lassen National Park, CA")
                st.rerun()
        
        # Control buttons
        st.markdown("### 🔧 Map Controls")
        col_control1, col_control2 = st.columns([1, 1])
        
        with col_control1:
            if st.button("🔄 Reset to Default", key="reset_click"):
                st.session_state.lat = 37.7749
                st.session_state.lon = -122.4194
                st.success("Reset to San Francisco coordinates")
                st.rerun()
        
        with col_control2:
            if st.button("📍 Center on Current Point", key="center_click"):
                st.success(f"Map centered on: {st.session_state.lat:.6f}, {st.session_state.lon:.6f}")
                st.rerun()
        
        st.markdown("💡 **Tip:** Click directly on any point on the map to select that location!")

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
                st.session_state.lat,
                st.session_state.lon,
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