# Data Collection Package

A comprehensive package for collecting various types of data for stability analysis, including terrain, soil, weather, and climate data.

## 📁 Package Structure

```
data_collection/
├── __init__.py                      # Package initialization
├── slope_data_collector.py          # Terrain and slope data collection
├── ssurgo_data_collector.py         # SSURGO soil data collection
├── cmip_data_collector.py           # CMIP6 climate data collection
├── cmip_model_predictor.py          # CMIP6 stability predictions
├── meteostat_data_collector.py      # Historical weather data collection
├── data_collection_orchestrator.py  # Main orchestrator
├── requirements.txt                 # Package dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For terrain data collection, you may need additional system dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# On macOS
brew install gdal

# On Windows
# Download GDAL from https://gdal.org/download.html
```

### Basic Usage

Run complete data collection:
```python
from data_collection_orchestrator import run_complete_data_collection

# Collect all available data types
result_df = run_complete_data_collection(
    input_file='your_data.csv',
    output_dir='collected_data'
)
```

## 📊 Data Collection Modules

### 1. Slope Data Collector (`slope_data_collector.py`)

Collects terrain and slope data using NED DEM tiles.

**Features:**
- Downloads NED (National Elevation Dataset) tiles
- Extracts terrain attributes: slope, aspect, curvature
- Creates terrain visualizations
- Handles batch processing

**Usage:**
```python
from slope_data_collector import SlopeDataCollector

collector = SlopeDataCollector()
features = collector.get_terrain_features_for_point(37.54189, -120.96683)
print(f"Slope: {features['slope_degrees']:.2f}°")
```

**Dependencies:** `leafmap`, `richdem`, `gdal`, `shapely`

### 2. SSURGO Data Collector (`ssurgo_data_collector.py`)

Collects soil data from USDA SSURGO database using SOAP API.

**Features:**
- Queries SSURGO SOAP API
- Extracts soil properties: texture, pH, bulk density, etc.
- Aggregates soil components
- Handles missing data gracefully

**Usage:**
```python
from ssurgo_data_collector import SSURGODataCollector

collector = SSURGODataCollector()
soil_data = collector.get_soil_data(37.0, -120.0)
print(f"Clay content: {soil_data['clay'].iloc[0]}%")
```

**Dependencies:** `requests`, `xmltodict`

### 3. CMIP Data Collector (`cmip_data_collector.py`)

Collects and processes CMIP6 climate model data.

**Features:**
- Loads CMIP6 NetCDF files
- Processes multiple scenarios (SSP126, SSP245, SSP370, SSP585)
- Calculates rolling precipitation statistics
- Handles historical and future data stitching

**Usage:**
```python
from cmip_data_collector import CMIPDataCollector

collector = CMIPDataCollector("data")
rolling_data = collector.process_all_scenarios()
value = collector.extract_data_at_location(rolling_data, 37.0, -120.0, datetime(2025, 1, 15), 'ssp245')
```

**Dependencies:** `xarray`, `cftime`, `netCDF4`

### 4. CMIP Model Predictor (`cmip_model_predictor.py`)

Predicts stability changes using CMIP6 climate forecasts.

**Features:**
- Trains Random Forest models on historical data
- Applies models to future climate scenarios
- Predicts stability changes from 2025-2100
- Generates comprehensive stability reports

**Usage:**
```python
from cmip_model_predictor import CMIPModelPredictor

predictor = CMIPModelPredictor(cmip_data)
results = predictor.train_stability_model(df, feature_columns)
stability_results = predictor.predict_stability_changes(df, scenario_datasets)
```

**Dependencies:** `scikit-learn`, `xarray`, `pandas`

### 5. Meteostat Data Collector (`meteostat_data_collector.py`)

Collects historical weather data using Meteostat API.

**Features:**
- Downloads historical weather data
- Calculates precipitation statistics
- Fills missing precipitation values
- Handles rate limiting

**Usage:**
```python
from meteostat_data_collector import MeteostatDataCollector

collector = MeteostatDataCollector()
precip_data = collector.get_precipitation_data(37.0, -120.0, datetime(2020, 1, 15))
print(f"Max 1-day precipitation: {precip_data['max_1_day_prcp']:.2f} mm")
```

**Dependencies:** `meteostat`, `tqdm`

## 🔧 Advanced Usage

### Using Individual Collectors

```python
from data_collection_orchestrator import DataCollectionOrchestrator

orchestrator = DataCollectionOrchestrator("output_dir")

# Collect specific data types
slope_df = orchestrator.collect_specific_data_type(df, 'slope')
soil_df = orchestrator.collect_specific_data_type(df, 'soil')
meteostat_df = orchestrator.collect_specific_data_type(df, 'meteostat')
```

### Weather Forecast Data Collection

```python
# Collect weather forecast data with climate dataset
forecast_df = orchestrator.collect_weather_forecast_data(
    df, 
    climate_file='path/to/climate_data.nc',
    parameter_name='pr'
)
```

### Data Validation

```python
# Validate collected data quality
validation_results = orchestrator.validate_collected_data(result_df)
print(validation_results)
```

## 📋 Data Types Collected

### Terrain Data
- **Slope (degrees)**: Terrain slope angle
- **Aspect**: Terrain aspect direction
- **Planform curvature**: Horizontal curvature
- **Profile curvature**: Vertical curvature

### Soil Data
- **Texture**: Clay, silt, sand percentages
- **pH**: Soil acidity/alkalinity
- **Bulk density**: Soil density
- **Available water capacity**: Water holding capacity
- **Hydraulic conductivity**: Water movement rate
- **Organic matter**: Organic content

### Weather Data
- **Historical precipitation**: Daily/monthly rainfall
- **Precipitation statistics**: Max/avg over different periods
- **Climate forecasts**: Future weather projections

## 🎯 Supported Data Sources

| Data Type | Source | API/Format | Coverage |
|-----------|--------|------------|----------|
| Terrain | NED (USGS) | GeoTIFF | United States |
| Soil | SSURGO (USDA) | SOAP API | United States |
| CMIP6 Climate | CMIP6 NetCDF | NetCDF | Global |
| Historical Weather | Meteostat | REST API | Global |

## 📈 Output Files

The data collection process generates several output files:

- `collected_data_complete.csv`: Complete dataset with all collected data
- `collection_summary.txt`: Summary of collection process
- `weather_forecast_data.csv`: Weather forecast data (if applicable)
- Terrain plots and visualizations (if requested)

## ⚠️ Important Notes

### Rate Limiting
- SSURGO API: No strict limits, but be respectful
- Meteostat API: Built-in rate limiting (0.1s delay)
- NED downloads: No limits, but large files

### Data Availability
- **Terrain data**: Available for most of the United States
- **Soil data**: Available for the United States only
- **Weather data**: Global coverage, but quality varies by location
- **Climate forecasts**: Requires specific NetCDF files

### Error Handling
All modules include comprehensive error handling:
- Graceful failure for missing data
- Warnings for API errors
- Fallback values for failed requests
- Detailed logging of issues

## 🛠️ Troubleshooting

### Common Issues

1. **GDAL not found**: Install GDAL system dependencies
2. **Meteostat import error**: Install with `pip install meteostat`
3. **SSURGO API errors**: Check internet connection and API availability
4. **Memory issues**: Process data in smaller batches

### Performance Tips

- Use batch processing for large datasets
- Enable progress bars with `tqdm`
- Cache downloaded data when possible
- Use appropriate rate limiting

## 📄 License

This package is provided as-is for educational and research purposes.

## 🤝 Contributing

To extend the data collection capabilities:

1. Add new collector modules following the existing pattern
2. Update the orchestrator to include new modules
3. Add appropriate error handling and validation
4. Update documentation and requirements 