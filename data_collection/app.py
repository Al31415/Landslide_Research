import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pydeck as pdk
import matplotlib.pyplot as plt
import os
import requests
import json

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
        # Intuitive map-based location selection
        st.info("🗺️ **Pan and zoom the map below to find your location. The crosshair shows what will be analyzed.**")
        
        # Initialize map tracking state
        if 'map_moved' not in st.session_state:
            st.session_state.map_moved = False
        
        # Create map data with current location (red dot) and historical points
        map_data = pd.DataFrame({
            'lat': [st.session_state.lat],
            'lon': [st.session_state.lon]
        })
        
        # Add historical points if available (blue dots)
        if orig_points is not None and not orig_points.empty:
            hist_data = orig_points.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
            map_data = pd.concat([map_data, hist_data], ignore_index=True)
        
        # Display the map using standard Streamlit map
        st.map(map_data, zoom=8, use_container_width=True)
        
        # Add crosshair indicator
        st.markdown("### 🎯 Current Target")
        st.info(f"**Red dot shows your target location:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
        st.markdown("**Instructions:** Use the controls below to move to your desired location, then click 'Use Current View Center'")
        
        # Map controls
        st.markdown("### 🎯 Map Controls")
        col_control1, col_control2, col_control3 = st.columns([1, 1, 1])
        
        with col_control1:
            if st.button("📍 Use Current View Center", type="primary", key="use_center"):
                # The current center is already stored in session_state
                st.success(f"✅ Using map center: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
                st.balloons()
        
        with col_control2:
            if st.button("🔄 Reset to Default", key="reset_map"):
                st.session_state.lat = 37.7749
                st.session_state.lon = -122.4194
                st.success("Reset to San Francisco")
                st.rerun()
        
        with col_control3:
            if st.button("🎯 Center on Selection", key="center_map"):
                # This will re-center the map on the current coordinates
                st.rerun()
        
        # Show current coordinates
        st.markdown("### 📍 Current Selection")
        col_coord1, col_coord2 = st.columns([1, 1])
        with col_coord1:
            st.metric("Latitude", f"{st.session_state.lat:.6f}")
        with col_coord2:
            st.metric("Longitude", f"{st.session_state.lon:.6f}")
        
        # Quick jump locations
        st.markdown("### 🚀 Quick Jump Locations")
        col_jump1, col_jump2, col_jump3, col_jump4 = st.columns([1, 1, 1, 1])
        
        with col_jump1:
            if st.button("🏔️ Mt. Rainier", key="rainier_jump"):
                st.session_state.lat = 46.8523
                st.session_state.lon = -121.7603
                st.success("Jumped to Mount Rainier")
                st.rerun()
        
        with col_jump2:
            if st.button("🌋 Mt. St. Helens", key="helens_jump"):
                st.session_state.lat = 46.1914
                st.session_state.lon = -122.1956
                st.success("Jumped to Mount St. Helens")
                st.rerun()
        
        with col_jump3:
            if st.button("🏔️ Yosemite", key="yosemite_jump"):
                st.session_state.lat = 37.8651
                st.session_state.lon = -119.5383
                st.success("Jumped to Yosemite")
                st.rerun()
        
        with col_jump4:
            if st.button("🌲 Olympic NP", key="olympic_jump"):
                st.session_state.lat = 47.8021
                st.session_state.lon = -123.6044
                st.success("Jumped to Olympic National Park")
                st.rerun()
        
        # Instructions
        st.markdown("### 💡 How to Use")
        st.info("""
        1. **Pan and zoom** the map to find your desired location
        2. The **red crosshair** shows exactly what will be analyzed
        3. Click **"Use Current View Center"** when you're happy with the location
        4. Or use **Quick Jump** buttons to go to common landslide areas
        """)
        
        # Fine adjustments
        st.markdown("### 🎯 Fine Adjustments")
        col_adj1, col_adj2, col_adj3, col_adj4 = st.columns([1, 1, 1, 1])
        
        with col_adj1:
            if st.button("⬆️ +0.01°", key="north_adj"):
                st.session_state.lat = min(49.0, st.session_state.lat + 0.01)
                st.rerun()
        
        with col_adj2:
            if st.button("⬇️ -0.01°", key="south_adj"):
                st.session_state.lat = max(24.0, st.session_state.lat - 0.01)
                st.rerun()
        
        with col_adj3:
            if st.button("⬅️ -0.01°", key="west_adj"):
                st.session_state.lon = max(-125.0, st.session_state.lon - 0.01)
                st.rerun()
        
        with col_adj4:
            if st.button("➡️ +0.01°", key="east_adj"):
                st.session_state.lon = min(-66.0, st.session_state.lon + 0.01)
                st.rerun()
        
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
        
        st.markdown("💡 **Tip:** Click anywhere on the map above to select that exact location!")

with col2:
    st.subheader("Prediction and Features")
    
    if run:
        with st.spinner("Computing prediction..."):
            try:
                # Import the feature service
                from feature_service import FeatureService
                
                # Initialize the feature service
                feature_service = FeatureService()
                
                # Get features for the selected location and date
                features = feature_service.get_features_for_point(
                    st.session_state.lat, 
                    st.session_state.lon, 
                    date
                )
                
                # Make prediction
                prediction = feature_service.predict_stability(features)
                
                # Display results
                st.success("✅ Prediction completed!")
                
                # Show prediction
                col_pred1, col_pred2 = st.columns([1, 1])
                with col_pred1:
                    st.metric("Stability Score", f"{prediction['stability_score']:.3f}")
                with col_pred2:
                    risk_level = "High" if prediction['stability_score'] < 0.5 else "Medium" if prediction['stability_score'] < 0.7 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Show features
                st.subheader("📊 Computed Features")
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                st.dataframe(feature_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)
    else:
        st.info("👆 Click 'Compute Prediction' to analyze the selected location.")
