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
except Exception:
	from ssurgo_data_collector import SSURGODataCollector
	from slope_data_collector import SlopeDataCollector
	from meteostat_data_collector import MeteostatDataCollector
	from cmip_data_collector import CMIPDataCollector

# Optional playground deps
try:
    import leafmap  # type: ignore
    _LEAFMAP_OK = True
except Exception:
    _LEAFMAP_OK = False

try:
    import richdem as _rd  # type: ignore
    _RICHDEM_OK = True
except Exception:
    _RICHDEM_OK = False

try:
    import rasterio as _rio  # type: ignore
    from rasterio.transform import rowcol as _rowcol  # type: ignore
    _RASTERIO_OK = True
except Exception:
    _RASTERIO_OK = False

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
                 model_path: str = str(Path(__file__).parent.parent / 'models' / 'best_model_RandomForest.joblib')):
		self.data_dir = data_dir
		self.model_path = model_path
		self._load_model()
		self.ssurgo = SSURGODataCollector()
		self.usgs = SlopeDataCollector()
		self.meteostat = MeteostatDataCollector()
		self.cmip = CMIPDataCollector(self.data_dir)
		self._rolling = None
		# Optional CSV cache for strict validation mode
		csv_path = Path(self.data_dir) / 'Corrected_Input_Data.csv'
		try:
			self._csv_df = pd.read_csv(csv_path) if csv_path.exists() else None
		except Exception:
			self._csv_df = None

	def _load_model(self) -> None:
		candidates = [
			Path(self.model_path),
			Path(__file__).parent.parent / 'models' / 'best_model_RandomForest.joblib',
			Path(__file__).parent / 'models' / 'best_model_RandomForest.joblib',
			Path.cwd() / 'models' / 'best_model_RandomForest.joblib',
		]
		for p in candidates:
			try:
				if p is not None and Path(p).exists():
					self.model = joblib.load(str(p))
					return
			except Exception:
				continue
		raise FileNotFoundError(f"Model file not found. Tried: {', '.join(str(p) for p in candidates)}")

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
						 progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
						 strict_from_csv: bool = False) -> Dict[str, Any]:
		features: Dict[str, Any] = {}
		units: Dict[str, str] = {}
		raw: Dict[str, Any] = {}

		def report(stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
			if progress_callback is not None:
				progress_callback(stage, message, data or {})

		# Optional strict CSV override for validation/playground parity
		if strict_from_csv and self._csv_df is not None and len(self._csv_df) > 0:
			try:
				row = self._match_csv_row(lat, lon, event_date)
				if row is not None:
					report('CSV Override', 'Using values from Corrected_Input_Data.csv for strict validation.', {})
					for k in REQUIRED_FEATURES:
						if k in row.index:
							v = row[k]
							try:
								features[k] = float(v)
							except Exception:
								features[k] = v
						# Define units for known fields
						units.update({
							'Slope From USGS Elevation Data': 'degrees',
							'Slope From SSURGO': 'degrees',
							'max_1_day_prcp': 'mm',
							'max_3_day_prcp': 'mm',
							'max_7_day_prcp': 'mm',
							'avg_30_day_prcp': 'mm/day',
							'avg_60_day_prcp': 'mm/day',
							'avg_90_day_prcp': 'mm/day',
							'avg_90_day_prcp_mean_flux': 'kg m^-2 s^-1',
							'avg_365_day_prcp_mean_flux': 'kg m^-2 s^-1',
							'Bulk Density': 'g/cm³',
							'Deepest Soil Horizon Layer': 'index',
						})
						features['_units'] = units
						features['_raw'] = raw
						return features
			except Exception:
				pass

		# SSURGO (align with playground/test2.py: first record, slope_h as 'slope', direct bulk_density)
		report('SSURGO', 'Querying SSURGO soil properties (playground style first-record)...')
		try:
			from ssurgo_data_collector import SSURGODataCollector as _S
			_sc = _S()
			soil_df = _sc.get_soil_data(lat, lon)
			if soil_df is not None and not soil_df.empty:
				primary = _sc.extract_primary_soil_properties(soil_df)
				row0 = primary.iloc[0] if not primary.empty else {}
				bulk_density = float(row0.get('bulk_density', 0.0)) if row0.get('bulk_density') is not None else 0.0
				slope_from_ssurgo = float(row0.get('slope', 0.0)) if row0.get('slope') is not None else 0.0
				if 'hzname' in primary.columns and not primary['hzname'].dropna().empty:
					deepest = float(_S._hzname_to_numeric(primary['hzname']).max())
					if pd.isna(deepest):
						deepest = 0.0
				else:
					deepest = 0.0
				features['Bulk Density'] = bulk_density
				features['Slope From SSURGO'] = slope_from_ssurgo
				features['Deepest Soil Horizon Layer'] = deepest
			else:
				features['Bulk Density'] = 0.0
				features['Slope From SSURGO'] = 0.0
				features['Deepest Soil Horizon Layer'] = 0.0
		except Exception:
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

		# USGS slope (robust internal collector to avoid runtime env differences)
		report('USGS NED 10m', 'Computing slope from internal collector (robust).')
		deg = np.nan
		try:
			terr = self.usgs.get_terrain_features_for_point(lat, lon)
			deg = float(terr.get('slope_degrees', np.nan))
		except Exception:
			deg = np.nan
		features['Slope From USGS Elevation Data'] = deg
		units['Slope From USGS Elevation Data'] = 'degrees'
		report('USGS NED 10m', 'USGS slope computed using Playground logic.', {
			'Slope From USGS Elevation Data (degrees)': features['Slope From USGS Elevation Data'],
		})

		# Meteostat (align with test3.py: interval-specific fetch + max/mean)
		report('Meteostat', 'Fetching precipitation data (playground intervals)...')
		try:
			from meteostat import Point, Daily  # type: ignore
			from datetime import timedelta as _td
			end = pd.to_datetime(event_date)
			location = Point(lat, lon)
			def _fetch(start, end):
				try:
					return Daily(location, start, end).fetch()
				except Exception:
					return pd.DataFrame()
			data_1 = _fetch(end - _td(days=1), end)
			data_3 = _fetch(end - _td(days=3), end)
			data_7 = _fetch(end - _td(days=7), end)
			data_14 = _fetch(end - _td(days=14), end)
			data_30 = _fetch(end - _td(days=30), end)
			data_60 = _fetch(end - _td(days=60), end)
			data_90 = _fetch(end - _td(days=90), end)
			data_365 = _fetch(end - _td(days=365), end)
			features['max_1_day_prcp'] = float(data_1['prcp'].max()) if ('prcp' in data_1 and not data_1.empty) else 0.0
			features['max_3_day_prcp'] = float(data_3['prcp'].max()) if ('prcp' in data_3 and not data_3.empty) else 0.0
			features['max_7_day_prcp'] = float(data_7['prcp'].max()) if ('prcp' in data_7 and not data_7.empty) else 0.0
			features['max_14_day_prcp'] = float(data_14['prcp'].max()) if ('prcp' in data_14 and not data_14.empty) else 0.0
			features['avg_30_day_prcp'] = float(data_30['prcp'].mean()) if ('prcp' in data_30 and not data_30.empty) else 0.0
			features['avg_60_day_prcp'] = float(data_60['prcp'].mean()) if ('prcp' in data_60 and not data_60.empty) else 0.0
			features['avg_90_day_prcp'] = float(data_90['prcp'].mean()) if ('prcp' in data_90 and not data_90.empty) else 0.0
			features['avg_365_day_prcp'] = float(data_365['prcp'].mean()) if ('prcp' in data_365 and not data_365.empty) else 0.0
			units.update({
				'max_1_day_prcp': 'mm', 'max_3_day_prcp': 'mm', 'max_7_day_prcp': 'mm', 'max_14_day_prcp': 'mm',
				'avg_30_day_prcp': 'mm/day', 'avg_60_day_prcp': 'mm/day', 'avg_90_day_prcp': 'mm/day', 'avg_365_day_prcp': 'mm/day'
			})
		except Exception:
			pr = self.meteostat.get_precipitation_data_playground_logic(lat, lon, end)
			for k, v in pr.items():
				val = float(v) if v is not None and not pd.isna(v) else 0.0
				features[k] = val
				units[k] = 'mm' if 'max_' in k else 'mm/day'
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
