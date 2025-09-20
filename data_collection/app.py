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
import folium
from streamlit_folium import st_folium
import shap

def _categorize_feature(feature_name: str) -> str:
    """Categorize features for better organization."""
    feature_lower = feature_name.lower()
    if 'slope' in feature_lower:
        return "🏔️ Terrain & Slope"
    elif any(term in feature_lower for term in ['prcp', 'precipitation']):
        return "🌧️ Precipitation"
    elif any(term in feature_lower for term in ['bulk', 'density', 'horizon', 'soil']):
        return "🌱 Soil Properties"
    elif 'flux' in feature_lower:
        return "🌊 Climate Model (CMIP6)"
    else:
        return "📊 Other Features"

st.set_page_config(page_title="Stability Predictor", layout="wide")

st.title("US Stability Predictor (CMIP + Meteostat + SSURGO + USGS)")
st.caption("🚀 Version 2.0 - Interactive Map Features | Last Updated: 2025-01-18")

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
        st.session_state.lat = 32.82426656  # First row latitude
    if 'lon' not in st.session_state:
        st.session_state.lon = -117.23500108  # First row longitude
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📍 Click on Map", "🔍 Search Location", "⌨️ Manual Entry"],
        horizontal=True
    )
    
    if input_method == "🔍 Search Location":
        # Search location input
        location_name = st.text_input("Enter location name:", placeholder="e.g., Mount Rainier, Washington")
        
        if st.button("🔍 Search", key="search_location"):
            if location_name:
                with st.spinner("Searching for location..."):
                    lat, lon, display_name = geocode_location(location_name)
                    if lat is not None and lon is not None:
                        st.session_state.lat = lat
                        st.session_state.lon = lon
                        st.session_state.last_searched_location = display_name
                        st.success(f"📍 Found: {display_name}")
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
        
        if st.button("📍 Set Location", key="set_location_manual"):
            st.session_state.lat = lat_input
            st.session_state.lon = lon_input
            st.success(f"📍 Location set to: {lat_input:.6f}, {lon_input:.6f}")
            st.rerun()
    
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
        # Use Folium + streamlit-folium for true click-to-select
        st.info("🗺️ Click anywhere on the map to select a location.")

        # Create a Folium map centered on current coords
        fmap = folium.Map(location=[float(st.session_state.lat), float(st.session_state.lon)], zoom_start=6, control_scale=True)
        folium.Marker(location=[float(st.session_state.lat), float(st.session_state.lon)]).add_to(fmap)

        # Render and capture interactions
        map_state = st_folium(fmap, height=420, key="folium_click_map")

        # Update on click
        if map_state and isinstance(map_state, dict) and map_state.get("last_clicked"):
            lat = map_state["last_clicked"].get("lat")
            lon = map_state["last_clicked"].get("lng")
            if lat is not None and lon is not None:
                st.session_state.lat = float(lat)
                st.session_state.lon = float(lon)
                st.success(f"📍 Selected: {st.session_state.lat:.6f}, {st.session_state.lon:.6f}")
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
            status_text.text("🔧 Initializing feature service...")
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
                status_text.text(f"🔄 {stage}: {message}")
                if data:
                    stage_details.json(data)
            
            # Get features for the selected location and date
            status_text.text("📊 Computing features...")
            features = feature_service.compute_features(
                float(st.session_state.lat),
                float(st.session_state.lon),
                datetime.combine(date, datetime.min.time()) if hasattr(date, 'year') else date,
                progress_callback=progress_callback
            )
            
            # Make prediction
            status_text.text("🤖 Making prediction...")
            progress_bar.progress(95)
            prediction = feature_service.predict(features)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Prediction completed!")
            stage_details.empty()
            
            with results_container:
                st.success("✅ Analysis Complete!")
                
                # Show prediction with enhanced styling
                st.subheader("🎯 Prediction Results")
                col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
                
                with col_pred1:
                    stability_score = prediction['stability_score']
                    st.metric("Stability Score", f"{stability_score:.4f}")
                
                with col_pred2:
                    risk_level = "🔴 High Risk" if stability_score < 0.3 else "🟡 Medium Risk" if stability_score < 0.7 else "🟢 Low Risk"
                    st.metric("Risk Assessment", risk_level)
                
                with col_pred3:
                    confidence = "High" if abs(stability_score - 0.5) > 0.3 else "Medium" if abs(stability_score - 0.5) > 0.15 else "Low"
                    st.metric("Prediction Confidence", confidence)
                
                # Enhanced feature display
                st.subheader("📊 Computed Features")
                
                # Filter out metadata
                display_features = {k: v for k, v in features.items() if not k.startswith('_')}
                
                # Create feature dataframe with units
                feature_data = []
                units_dict = features.get('_units', {})
                
                for feature_name, value in display_features.items():
                    unit = units_dict.get(feature_name, '')
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        formatted_value = f"{value:.6f}" if abs(value) < 1 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
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
                    with st.expander(f"📋 {category} Features", expanded=True):
                        cat_features = feature_df[feature_df['Category'] == category][['Feature', 'Value', 'Unit']]
                        st.dataframe(cat_features, use_container_width=True, hide_index=True)
                
                # SHAP Values Analysis
                st.subheader("🔍 Feature Importance (SHAP Analysis)")
                
                try:
                    # Prepare feature vector for SHAP
                    from feature_service import REQUIRED_FEATURES
                    X = pd.DataFrame([{k: display_features.get(k, np.nan) for k in REQUIRED_FEATURES}])
                    X = X.fillna(0)
                    
                    # Create SHAP explainer
                    explainer = shap.Explainer(feature_service.model)
                    shap_values = explainer(X)
                    
                    # Display SHAP values
                    col_shap1, col_shap2 = st.columns([1, 1])
                    
                    with col_shap1:
                        st.markdown("**SHAP Feature Contributions:**")
                        
                        # Create SHAP summary
                        shap_data = []
                        values_array = np.asarray(shap_values.values)
                        for i, feature in enumerate(REQUIRED_FEATURES):
                            raw_val = values_array[0, i] if values_array.size > 0 else 0.0
                            # Coerce to scalar float
                            if np.isscalar(raw_val):
                                val = float(raw_val)
                            else:
                                val = float(np.asarray(raw_val).ravel()[0])
                            shap_data.append({
                                'Feature': feature,
                                'SHAP Value': f"{val:.6f}",
                                'Impact': "🔴 Increases Risk" if val > 0 else "🟢 Decreases Risk" if val < 0 else "⚪ Neutral",
                                'abs_value': abs(val)
                            })
                        
                        shap_df = pd.DataFrame(shap_data).sort_values(by='abs_value', ascending=False)
                        st.dataframe(shap_df[['Feature','SHAP Value','Impact']], use_container_width=True, hide_index=True)
                
                    with col_shap2:
                        st.markdown("**SHAP Waterfall Plot:**")
                        
                        # Create SHAP waterfall plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        # Ensure SHAP values indexable as expected
                        shap.waterfall_plot(shap_values[0], show=False)
                        st.pyplot(fig)
                        plt.close()
                
                except Exception as shap_error:
                    st.warning(f"SHAP analysis failed: {shap_error}")
                    
                    # Fallback: Simple feature importance
                    st.markdown("**Feature Values (Fallback Display):**")
                    importance_data = []
                    for feature_name, value in display_features.items():
                        importance_data.append({
                            'Feature': feature_name,
                            'Value': f"{value:.6f}" if isinstance(value, (int, float)) and not pd.isna(value) else str(value)
                        })
                    
                    importance_df = pd.DataFrame(importance_data)
                    st.dataframe(importance_df, use_container_width=True, hide_index=True)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            stage_details.empty()
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
    else:
        st.info("👆 Click 'Compute Prediction' to analyze the selected location.")

 
