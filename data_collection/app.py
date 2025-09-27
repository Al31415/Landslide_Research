import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pydeck as pdk
import matplotlib.pyplot as plt
import os
import requests
import json
import streamlit.components.v1 as components
import shap

def _categorize_feature(feature_name: str) -> str:
    """Categorize features for better organization."""
    feature_lower = feature_name.lower()
    if 'slope' in feature_lower:
        return "Terrain & Slope"
    elif any(term in feature_lower for term in ['prcp', 'precipitation']):
        return "Precipitation"
    elif any(term in feature_lower for term in ['bulk', 'density', 'horizon', 'soil']):
        return "Soil Properties"
    elif 'flux' in feature_lower:
        return "Climate Model (CMIP6)"
    else:
        return "Other Features"

def _format_feature_value(feature_name: str, value) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "NA"
        v = float(value)
        # CMIP mean flux values are very small; show scientific notation
        if 'mean_flux' in feature_name:
            return f"{v:.2e}"
        # Precipitation metrics in mm – 2 decimals are enough
        if 'prcp' in feature_name:
            return f"{v:.2f}"
        # General numeric formatting
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.6f}"
    except Exception:
        return str(value)

st.set_page_config(page_title="Stability Predictor", layout="wide")

st.title("US Stability Predictor (CMIP + Meteostat + SSURGO + USGS)")


def geocode_location(location_name):
    """
    Geocode a location name to get latitude and longitude coordinates.
    Uses OpenStreetMap Nominatim API (free, no API key required).
    """
    try:
        # Use Nominatim API for geocoding
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'Stability Predictor App'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon, data[0]['display_name']
        else:
            return None, None, None
            
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        return None, None, None

# Load historical data points
@st.cache_data
def load_historical_data():
    try:
        # Try to load the historical data
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Corrected_Input_Data.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return df[['Latitude', 'Longitude']].dropna()
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")
        return pd.DataFrame()

orig_points = load_historical_data()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Location and Date")
    
    # Initialize session state for coordinates (using first row from Corrected_Input_Data.csv)
    if 'lat' not in st.session_state:
        try:
            if orig_points is not None and not orig_points.empty:
                st.session_state.lat = float(orig_points.iloc[0]['Latitude'])
            else:
                st.session_state.lat = 32.82426656
        except Exception:
            st.session_state.lat = 32.82426656
    if 'lon' not in st.session_state:
        try:
            if orig_points is not None and not orig_points.empty:
                st.session_state.lon = float(orig_points.iloc[0]['Longitude'])
            else:
                st.session_state.lon = -117.23500108
        except Exception:
            st.session_state.lon = -117.23500108
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📍 Click on Map", "🔍 Search Location", "⌨️ Manual Entry"],
        horizontal=True
    )
    
    if input_method == "🔍 Search Location":
        # Search location input
        location_name = st.text_input("Enter location name:", placeholder="e.g., Mount Rainier, Washington")
        
        if st.button("Search", key="search_location"):
            if location_name:
                with st.spinner("Searching for location..."):
                    lat, lon, display_name = geocode_location(location_name)
                    if lat is not None and lon is not None:
                        st.session_state.lat = lat
                        st.session_state.lon = lon
                        st.session_state.last_searched_location = display_name
                        st.success(f"Found: {display_name}")
                        st.rerun()
                    else:
                        st.error("Location not found. Please try a different search term.")
            else:
                st.warning("Please enter a location name.")
    
    elif input_method == "⌨️ Manual Entry":
        # Manual coordinate input
        col_lat, col_lon = st.columns([1, 1])
        
        with col_lat:
            lat_input = st.number_input(
                "Latitude", 
                min_value=-90.0, 
                max_value=90.0, 
                value=float(st.session_state.lat), 
                step=0.000001,
                format="%.6f"
            )
        
        with col_lon:
            lon_input = st.number_input(
                "Longitude", 
                min_value=-180.0, 
                max_value=180.0, 
                value=float(st.session_state.lon), 
                step=0.000001,
                format="%.6f"
            )
        
        if st.button("Set Location", key="set_location_manual"):
            st.session_state.lat = lat_input
            st.session_state.lon = lon_input
            st.success(f"Location set to: {lat_input:.6f}, {lon_input:.6f}")
            st.rerun()
    
    # Display current coordinates with better formatting
    st.markdown("### Current Location")
    col_coord1, col_coord2 = st.columns([1, 1])
    with col_coord1:
        st.metric("Latitude", f"{st.session_state.lat:.6f}")
    with col_coord2:
        st.metric("Longitude", f"{st.session_state.lon:.6f}")
    
    # Add a quick location info display
    if hasattr(st.session_state, 'last_searched_location'):
        st.info(f"Last searched: {st.session_state.last_searched_location}")
    
    # Event date input (default to first row of Corrected_Input_Data.csv)
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Corrected_Input_Data.csv')
        if 'default_event_date' not in st.session_state:
            if os.path.exists(data_path):
                _df0 = pd.read_csv(data_path, nrows=1)
                st.session_state.default_event_date = pd.to_datetime(_df0['event_date'].iloc[0]).date()
            else:
                st.session_state.default_event_date = datetime(2025, 1, 15).date()
    except Exception:
        st.session_state.default_event_date = datetime(2025, 1, 15).date()
    date = st.date_input("Event Date", value=st.session_state.default_event_date)
    
    # Compute button
    run = st.button("Compute Prediction", type="primary")

    # Map interface - different for each input method
    st.subheader("Map View")

    if input_method == "🔍 Search Location":
        # Simple visualization map for search results
        st.info("Map showing your searched location and historical data points.")
        
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
        st.info("Map showing your manually entered coordinates and historical data points.")
        
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
        # Use Folium + streamlit-folium for true click-to-select
        st.info("Click anywhere on the map to select a location.")

        # Local import to avoid linter missing-import warning when not installed in dev env
        try:
            import folium  # type: ignore
            from streamlit_folium import st_folium  # type: ignore
        except Exception:
            st.error("Folium components are unavailable. Please ensure folium and streamlit-folium are installed.")
            st.stop()

        # Manage a dynamic key to force reliable re-renders if needed
        if 'folium_key' not in st.session_state:
            st.session_state.folium_key = 0

        with st.spinner("Loading map..."):
            # Create a Folium map centered on current coords
            fmap = folium.Map(
                location=[float(st.session_state.lat), float(st.session_state.lon)],
                zoom_start=6,
                control_scale=True,
                prefer_canvas=True,
                tiles=None
            )
            folium.TileLayer(
                tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr='© OpenStreetMap contributors'
            ).add_to(fmap)
            folium.Marker(location=[float(st.session_state.lat), float(st.session_state.lon)]).add_to(fmap)

            # Render and capture interactions
            map_state = st_folium(
                fmap,
                height=420,
                key=f"folium_click_map_{st.session_state.folium_key}",
                returned_objects=["last_clicked"]
            )

        cols_reload = st.columns([1, 3])
        with cols_reload[0]:
            if st.button("Reload map", key="reload_folium"):
                st.session_state.folium_key += 1
                st.rerun()

        # Update on click
        if map_state and isinstance(map_state, dict) and map_state.get("last_clicked"):
            lat = map_state["last_clicked"].get("lat")
            lon = map_state["last_clicked"].get("lng")
            if lat is not None and lon is not None:
                st.session_state.lat = float(lat)
                st.session_state.lon = float(lon)
                st.success(f"Selected: {st.session_state.lat:.6f}, {st.session_state.lon:.6f}")
                st.rerun()

with col2:
    st.subheader("Prediction and Features")
    
    if run:
        # Create progress tracking containers
        progress_container = st.container()
        results_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            stage_details = st.empty()
        
        try:
            # Import the feature service
            from feature_service import FeatureService
            
            # Initialize the feature service
            status_text.text("Initializing feature service...")
            progress_bar.progress(10)
            feature_service = FeatureService()
            
            # Progress tracking callback
            def progress_callback(stage: str, message: str, data: dict):
                stage_progress = {
                    'SSURGO': 30,
                    'USGS NED 10m': 50,
                    'Meteostat': 70,
                    'CMIP6 CESM2': 85
                }
                progress_bar.progress(stage_progress.get(stage, 90))
                status_text.text(f"{stage}: {message}")
                if data:
                    stage_details.json(data)
            
            # Get features for the selected location and date
            status_text.text("Computing features...")
            features = feature_service.compute_features(
                float(st.session_state.lat),
                float(st.session_state.lon),
                datetime.combine(date, datetime.min.time()) if hasattr(date, 'year') else date,
                progress_callback=progress_callback,
                strict_from_csv=True
            )
            
            # Make prediction
            status_text.text("Making prediction...")
            progress_bar.progress(95)
            prediction = feature_service.predict(features)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("Prediction completed.")
            stage_details.empty()
            
            with results_container:
                st.success("Analysis Complete")
                
                # Show prediction with enhanced styling
                st.subheader("Prediction Results")
                col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
                
                with col_pred1:
                    stability_score = prediction['stability_score']
                    st.metric("Stability Score", f"{stability_score:.4f}")
                
                with col_pred2:
                    risk_level = "High Risk" if stability_score < 0.3 else "Medium Risk" if stability_score < 0.7 else "Low Risk"
                    st.metric("Risk Assessment", risk_level)
                
                with col_pred3:
                    confidence = "High" if abs(stability_score - 0.5) > 0.3 else "Medium" if abs(stability_score - 0.5) > 0.15 else "Low"
                    st.metric("Prediction Confidence", confidence)
                
                # Enhanced feature display
                st.subheader("Computed Features")
                
                # Filter out metadata
                display_features = {k: v for k, v in features.items() if not k.startswith('_')}
                
                # Create feature dataframe with units
                feature_data = []
                units_dict = features.get('_units', {})
                
                for feature_name, value in display_features.items():
                    unit = units_dict.get(feature_name, '')
                    formatted_value = _format_feature_value(feature_name, value)
                    
                    feature_data.append({
                        'Feature': feature_name,
                        'Value': formatted_value,
                        'Unit': unit,
                        'Category': _categorize_feature(feature_name)
                    })
                
                feature_df = pd.DataFrame(feature_data)
                
                # Display features by category
                categories = feature_df['Category'].unique()
                for category in sorted(categories):
                    with st.expander(f"{category} Features", expanded=True):
                        cat_features = feature_df[feature_df['Category'] == category][['Feature', 'Value', 'Unit']]
                        st.dataframe(cat_features, use_container_width=True, hide_index=True)
                
                # SHAP Values Analysis
                st.subheader("Feature Importance (SHAP Analysis)")

                try:
                    # Prepare feature vector for SHAP
                    from feature_service import REQUIRED_FEATURES
                    X = pd.DataFrame([{k: display_features.get(k, np.nan) for k in REQUIRED_FEATURES}]).fillna(0)

                    # Create SHAP explainer and compute values
                    explainer = shap.Explainer(feature_service.model)
                    shap_values = explainer(X)

                    values_array = np.asarray(shap_values.values)
                    if values_array.size == 0:
                        raise ValueError("Empty SHAP values returned")
                    if values_array.ndim == 3 and values_array.shape[2] > 1:
                        contribs = values_array[0, :, 1]
                        exp = shap_values[:, :, 1][0]
                    elif values_array.ndim == 2:
                        contribs = values_array[0, :]
                        exp = shap_values[0]

                    col_shap1, col_shap2 = st.columns([1, 1])

                    with col_shap1:
                        st.markdown("SHAP Feature Contributions")
                        shap_rows = []
                        max_len = min(len(REQUIRED_FEATURES), len(contribs))
                        for i in range(max_len):
                            feature = REQUIRED_FEATURES[i]
                            val = float(contribs[i])
                            shap_rows.append({
                    'Feature': feature,
                                'SHAP Value': f"{val:.6f}",
                                'Impact': ("Increases Risk" if val > 0 else ("Decreases Risk" if val < 0 else "Neutral")),
                                'abs_value': abs(val)
                            })
                        shap_df = pd.DataFrame(shap_rows).sort_values(by='abs_value', ascending=False)
                        st.dataframe(shap_df[['Feature','SHAP Value','Impact']], use_container_width=True, hide_index=True)

                    with col_shap2:
                        st.markdown("SHAP Waterfall Plot")
                        try:
                            fig = plt.figure(figsize=(10, 8))
                            shap.plots.waterfall(exp, show=False)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e2:
                            st.warning(f"Waterfall plot failed: {e2}")
                            st.text("SHAP values shape: " + str(values_array.shape))

                except Exception as e:
                    st.warning(f"SHAP analysis failed: {e}")
                    st.markdown("Feature Values (Fallback Display)")
                    fallback_rows = []
                    for feature_name, value in display_features.items():
                        fallback_rows.append({
                            'Feature': feature_name,
                            'Value': _format_feature_value(feature_name, value)
                        })
                    st.dataframe(pd.DataFrame(fallback_rows), use_container_width=True, hide_index=True)

                # Interactive 3D (auto-rendered)
                collector = None
                try:
                    from slope_data_collector import SlopeDataCollector
                    # Reuse a single collector instance across session so debug carries over
                    if 'slope_collector' not in st.session_state:
                        st.session_state.slope_collector = SlopeDataCollector()
                    collector = st.session_state.slope_collector
                    with st.spinner("Building interactive 3D view (Plotly)..."):
                        fig_int, res_label = collector.build_interactive_3d(
                            lat=float(st.session_state.lat),
                            lon=float(st.session_state.lon),
                            half_side_m=400
                        )
                    fig_int.update_layout(height=520)
                    st.info(f"Interactive 3D resolution used: {res_label}")
                    st.plotly_chart(fig_int, use_container_width=True)
                except Exception as inter_err:
                    st.warning(f"Interactive 3D failed: {inter_err}")
                # Always show diagnostics block (whether 3D rendered or not)
                try:
                    if collector is not None:
                        with st.expander("Diagnostics", expanded=False):
                            st.markdown("Selected Location")
                            st.json({
                                'lat': float(st.session_state.lat),
                                'lon': float(st.session_state.lon),
                            })
                            # Show high-level last states
                            if hasattr(collector, '_last_debug_download'):
                                st.markdown("Download (Last)")
                                st.json(getattr(collector, '_last_debug_download'))
                            if hasattr(collector, '_last_debug_usgs'):
                                st.markdown("USGS DEM (Slope Feature)")
                                st.json(getattr(collector, '_last_debug_usgs'))
                            if hasattr(collector, '_last_dem_read'):
                                st.markdown("DEM Read (Last)")
                                st.json(getattr(collector, '_last_dem_read'))
                            if hasattr(collector, '_last_debug_3d'):
                                st.markdown("3D Renderer (Last)")
                                _dbg3d = getattr(collector, '_last_debug_3d')
                                st.json(_dbg3d)
                                try:
                                    html_path = _dbg3d.get('html_path') if isinstance(_dbg3d, dict) else None
                                    if html_path and os.path.exists(html_path):
                                        st.caption(f"Saved 3D HTML: {html_path}")
                                        with open(html_path, 'r', encoding='utf-8') as _f:
                                            _html = _f.read()
                                        st.download_button(
                                            label="Download 3D HTML",
                                            data=_html,
                                            file_name=os.path.basename(html_path),
                                            mime="text/html",
                                        )
                                        components.html(_html, height=600, scrolling=True)
                                except Exception as _emb_err:
                                    st.caption(f"Preview unavailable: {_emb_err}")
                            # Full debug log with all calls
                            if hasattr(collector, '_debug_log'):
                                st.markdown("### Full Debug Log")
                                try:
                                    st.json(collector._debug_log)
                                except Exception:
                                    # Fallback pretty print
                                    st.text(str(collector._debug_log))
                except Exception:
                    pass

                # AI-Assisted Slope/Soil Summary (auto, uses OpenAI 4o if key present)
                try:
                    import os as _os
                    openai_key = _os.environ.get("OPENAI_API_KEY")
                    if openai_key:
                        from openai import OpenAI
                        client = OpenAI(api_key=openai_key)

                        summary_payload = {
                            "lat": float(st.session_state.lat),
                            "lon": float(st.session_state.lon),
                            "event_date": str(date),
                            "prediction": {
                                "stability_score": float(stability_score),
                                "risk_level": risk_level,
                            },
                            "features": {k: display_features.get(k) for k in display_features.keys()},
                            "units": units_dict,
                        }

                        # Load feature description context if present
                        extra_context = ""
                        try:
                            import docx
                            _doc_path = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'Feature Descriptions.docx')
                            if _os.path.exists(_doc_path):
                                _doc = docx.Document(_doc_path)
                                extra_context = "\n\nFeature Descriptions:\n" + "\n".join(p.text for p in _doc.paragraphs if p.text.strip())
                        except Exception:
                            pass

                        sys_prompt = (
                            "You are a geotechnical assistant. Infer terrain and soil characteristics near the given coordinates "
                            "based on the numeric features and risk prediction. Explain likely soil/rock type, drainage, slope stability factors, "
                            "and data caveats. Be concise, actionable, and avoid overstating certainty."
                        )
                        user_prompt = (
                            "Using this context, summarize what the slope/soil are likely like, including inferred soil/rock type if possible, "
                            "and any geomorphological cues that would matter for stability. "
                            "Note: 'Slope From SSURGO' is soil-map derived and may differ from 'Slope From USGS Elevation Data' which is a pixel-level DEM (richdem) slope."
                        )

                        st.subheader("AI Slope & Soil Summary")
                        with st.spinner("Generating AI summary..."):
                            resp = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": f"Context: {summary_payload}{extra_context}\n\n{user_prompt}"},
                                ],
                                temperature=0.4,
                                max_tokens=500,
                            )
                        ai_text = resp.choices[0].message.content if resp and getattr(resp, 'choices', None) else ""
                        if ai_text:
                            st.markdown(ai_text)
                            st.caption("Model: gpt-4o")
                except Exception as ai_err:
                    st.warning(f"AI summary failed: {ai_err}")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            stage_details.empty()
            st.error(f"Prediction failed: {e}")
            st.exception(e)
    else:
        st.info("Click 'Compute Prediction' to analyze the selected location.")

 


