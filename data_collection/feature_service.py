"""
Feature Service
Computes model-ready features for a given (lat, lon, date) and returns prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import joblib

try:
	from .ssurgo_data_collector import SSURGODataCollector
	from .slope_data_collector import SlopeDataCollector
	from .meteostat_data_collector import MeteostatDataCollector
	from .cmip_data_collector import CMIPDataCollector
except ImportError:
	from ssurgo_data_collector import SSURGODataCollector
	from slope_data_collector import SlopeDataCollector
	from meteostat_data_collector import MeteostatDataCollector
	from cmip_data_collector import CMIPDataCollector

REQUIRED_FEATURES = [
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
	'Deepest Soil Horizon Layer'
]

class FeatureService:
	def __init__(self, data_dir: str = str(Path(__file__).parent.parent / 'data'),
	             model_path: str = str(Path(__file__).parent.parent / 'models' / 'best_model.joblib')):
		self.data_dir = data_dir
		self.model_path = model_path
		self._load_model()
		self.ssurgo = SSURGODataCollector()
		self.usgs = SlopeDataCollector()
		self.meteostat = MeteostatDataCollector()
		self.cmip = CMIPDataCollector(self.data_dir)
		self._rolling = None

	def _load_model(self) -> None:
		self.model = joblib.load(self.model_path)

	def _ensure_cmip(self) -> None:
		if self._rolling is None:
			self._rolling = self.cmip.process_all_scenarios()

	@staticmethod
	def _hzname_to_numeric(hzname_series: pd.Series) -> pd.Series:
		s = hzname_series.fillna("").astype(str)
		s = s.str.replace('[^A-Z]', '', regex=True)
		s = s.str.replace('BE', '3.5', regex=False)
		s = s.str.replace('BC', '4.5', regex=False)
		s = s.str.replace('AC', '3.5', regex=False)
		s = s.str.replace('EB', '3.5', regex=False)
		s = s.str.replace('AB', '3', regex=False)
		s = s.str.replace('AE', '2.5', regex=False)
		s = s.str.replace('O', '1', regex=False)
		s = s.str.replace('H', '1', regex=False)
		s = s.str.replace('A', '2', regex=False)
		s = s.str.replace('E', '3', regex=False)
		s = s.str.replace('B', '4', regex=False)
		s = s.str.replace('C', '5', regex=False)
		return pd.to_numeric(s, errors='coerce')

	def compute_features(self, lat: float, lon: float, event_date: datetime,
						 progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
		features: Dict[str, Any] = {}
		units: Dict[str, str] = {}
		raw: Dict[str, Any] = {}

		def report(stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
			if progress_callback is not None:
				progress_callback(stage, message, data or {})

		# SSURGO (Playground logic)
		report('SSURGO', 'Querying SSURGO soil properties using Playground logic...')
		soil_features = self.ssurgo.get_soil_features_for_point(lat, lon)
		features['Bulk Density'] = soil_features['bulk_density']
		features['Slope From SSURGO'] = soil_features['slope_from_ssurgo']
		features['Deepest Soil Horizon Layer'] = soil_features['Deepest_Soil_Horizon_Layer']
		units['Bulk Density'] = 'g/cm³'
		units['Slope From SSURGO'] = 'degrees'
		units['Deepest Soil Horizon Layer'] = 'index (mapped from hzname)'
		report('SSURGO', 'SSURGO soil properties extracted using Playground logic.', {
			'Bulk Density (g/cm³)': features['Bulk Density'],
			'Slope From SSURGO (degrees)': features['Slope From SSURGO'],
			'Deepest Soil Horizon Layer': features['Deepest Soil Horizon Layer'],
		})

		# USGS (Playground logic; RichDEM slope in degrees)
		report('USGS NED 10m', 'Computing slope from elevation data using Playground logic...')
		terr = self.usgs.get_terrain_features_for_point(lat, lon)
		deg = float(terr.get('slope_degrees', np.nan))
		features['Slope From USGS Elevation Data'] = deg
		units['Slope From USGS Elevation Data'] = 'degrees'
		report('USGS NED 10m', 'USGS slope computed using Playground logic.', {
			'Slope From USGS Elevation Data (degrees)': features['Slope From USGS Elevation Data'],
		})

		# Meteostat (Playground logic)
		report('Meteostat', 'Fetching precipitation data using Playground logic...')
		pr = self.meteostat.get_precipitation_data_playground_logic(lat, lon, event_date)
		for k, v in pr.items():
			val = float(v) if v is not None and not pd.isna(v) else np.nan
			if pd.notna(val) and abs(val) < 1e-6:
				val = 0.0
			features[k] = val
			if 'max_' in k:
				units[k] = 'mm'
			elif 'avg_' in k:
				units[k] = 'mm/day'
		report('Meteostat', 'Meteostat precipitation data fetched using Playground logic.', {
			'max_1_day_prcp (mm)': features.get('max_1_day_prcp'),
			'max_3_day_prcp (mm)': features.get('max_3_day_prcp'),
			'max_7_day_prcp (mm)': features.get('max_7_day_prcp'),
			'avg_30_day_prcp (mm/day)': features.get('avg_30_day_prcp'),
			'avg_60_day_prcp (mm/day)': features.get('avg_60_day_prcp'),
			'avg_90_day_prcp (mm/day)': features.get('avg_90_day_prcp'),
		})

		# CMIP (optional; skip if unavailable)
		try:
			report('CMIP6 CESM2', 'Loading CMIP6 scenarios and computing mean flux proxies...')
			self._ensure_cmip()
			cmip_flux = self.cmip.compute_mean_flux_features(self._rolling, lat, lon, event_date, scenario='ssp245', windows_days=[90, 365])
			for k, v in cmip_flux.items():
				val = float(v) if v is not None and not pd.isna(v) else np.nan
				if pd.notna(val) and abs(val) < 1e-6:
					val = 0.0
				features[k] = val
				units[k] = 'kg m^-2 s^-1'
			report('CMIP6 CESM2', 'CMIP6 mean flux proxies computed.', {
				'avg_90_day_prcp_mean_flux (kg m^-2 s^-1)': features.get('avg_90_day_prcp_mean_flux'),
				'avg_365_day_prcp_mean_flux (kg m^-2 s^-1)': features.get('avg_365_day_prcp_mean_flux'),
			})
		except Exception as e:
			report('CMIP6 CESM2', f'Skipping CMIP6 features: {e}', {})

		features['_units'] = units
		features['_raw'] = raw
		return features

	def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
		X = pd.DataFrame([{k: features.get(k, np.nan) for k in REQUIRED_FEATURES}])
		X = X.fillna(0)
		y_pred = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X)
		score = float(y_pred[0]) if isinstance(y_pred, np.ndarray) else float(y_pred)
		return {
			'stability_score': score,
			'stable': bool(score >= 0.5),
		} 
