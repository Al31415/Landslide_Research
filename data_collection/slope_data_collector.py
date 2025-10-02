"""
Slope Data Collector Module
Collects terrain and slope data using NED DEM tiles.
"""

import math
import os
import random
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import geopy.distance
import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import richdem as rd
    RICHDEM_AVAILABLE = True
except ImportError:
    RICHDEM_AVAILABLE = False
    print("Warning: richdem not available. Using fallback slope calculation.")
import shapefile
try:
    from osgeo import gdal  # type: ignore
    GDAL_AVAILABLE = True
except Exception:
    GDAL_AVAILABLE = False
from shapely.geometry import Point, Polygon
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    import rasterio as rio  # type: ignore
    from rasterio.transform import rowcol as rio_rowcol  # type: ignore
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False
import plotly.graph_objects as go


class SlopeDataCollector:
    """
    Collect slope and terrain data for given coordinates.
    
    This module downloads NED DEM tiles and extracts terrain attributes
    including slope, aspect, and curvature for specified locations.
    """

    def __init__(
        self,
        shapefile_path: Optional[str] = None,
        max_distance_km: float = 50,
        lat_range: Tuple[float, float] = (25, 50),
        lon_range: Tuple[float, float] = (-130, -60),
        ned_resolution_m: int = 10,  # 10 m for 1/3-arc-second NED
    ):
        """
        Initialize the slope data collector.
        
        Args:
            shapefile_path: Path to US border shapefile (optional)
            max_distance_km: Maximum distance for point generation
            lat_range: Latitude range for data collection
            lon_range: Longitude range for data collection
            ned_resolution_m: NED resolution in meters
        """
        self.max_distance_km = max_distance_km
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.ned_resolution_m = ned_resolution_m
        
        # Initialize state polygons if shapefile provided
        self.state_polygons = {}
        if shapefile_path:
            try:
                self.state_polygons = self._get_us_border_polygon(shapefile_path)
            except Exception as e:
                warnings.warn(f"Could not load shapefile: {e}")

        # Debug log container
        self._debug_log = {
            'init': {
                'richdem_available': RICHDEM_AVAILABLE,
                'gdal_available': GDAL_AVAILABLE,
                'rasterio_available': RASTERIO_AVAILABLE,
                'ned_resolution_m': self.ned_resolution_m,
            },
            'calls': [],
        }

    def _debug_record(self, method: str, step: str, data: Dict[str, Any]) -> None:
        """Record debug information for a method step."""
        call_entry = {
            'method': method,
            'step': step,
            'timestamp': str(pd.Timestamp.now()),
            'data': data,
        }
        self._debug_log['calls'].append(call_entry)
        # Cap debug log to avoid unbounded memory growth
        try:
            if len(self._debug_log['calls']) > 300:
                self._debug_log['calls'] = self._debug_log['calls'][-300:]
        except Exception:
            pass

    def _read_cropped_dem_and_slope(self, dem_file: str, lat: float, lon: float,
                                    half_side_m: int,
                                    pad_px: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficiently read a cropped DEM window around (lat, lon) and compute slope
        on a slightly larger padded window so that the interior crop's slope is
        identical to computing slope on the full raster (local operator).

        Returns (dem_crop, slope_crop) where each is (H, W) covering +/- half_side_m
        in meters around (lat, lon). The slope is in degrees.
        """
        self._debug_record('_read_cropped_dem_and_slope', 'start', {
            'dem_file': str(dem_file), 'lat': float(lat), 'lon': float(lon),
            'half_side_m': int(half_side_m), 'pad_px': int(pad_px),
        })
        # Target pixel radius for interior crop
        pixels = int(max(1, round(half_side_m / float(self.ned_resolution_m))))

        # Prefer rasterio for precise windowed IO
        if RASTERIO_AVAILABLE:
            try:
                import rasterio as _rio_local  # type: ignore
                from rasterio.windows import Window  # type: ignore
                with _rio_local.open(str(dem_file)) as ds_r:
                    tr = ds_r.transform
                    # Build GDAL-like tuple for shared helpers
                    gt = (tr.c, tr.a, tr.b, tr.f, tr.d, tr.e)
                    w_full, h_full = ds_r.width, ds_r.height
                    # Pixel center
                    r_c, c_c = rio_rowcol(tr, lon, lat)
                    px_c, py_c = int(c_c), int(r_c)
                    # Window with padding for gradient locality
                    pad = int(max(1, pad_px))
                    xmin = max(0, px_c - (pixels + pad))
                    xmax = min(w_full, px_c + (pixels + pad))
                    ymin = max(0, py_c - (pixels + pad))
                    ymax = min(h_full, py_c + (pixels + pad))
                    win = Window(col_off=xmin, row_off=ymin,
                                 width=max(1, xmax - xmin), height=max(1, ymax - ymin))
                    dem_win = ds_r.read(1, window=win).astype(np.float32)
                    nodata_val = None
                    try:
                        nodata_val = ds_r.nodata
                        if nodata_val is not None:
                            dem_win = np.where(dem_win == nodata_val, np.nan, dem_win)
                    except Exception:
                        pass
                    # Sanitize extreme sentinels
                    try:
                        dem_win = np.where(dem_win <= -1e5, np.nan, dem_win)
                    except Exception:
                        pass
                    # Replace NaNs with local mean to keep slope contiguous
                    if not np.isfinite(dem_win).any():
                        dem_win = np.zeros_like(dem_win, dtype=np.float32)
                    else:
                        mean_val = float(np.nanmean(dem_win))
                        dem_win = np.where(np.isfinite(dem_win), dem_win, mean_val)

                    # Compute slope on padded window
                    if RICHDEM_AVAILABLE:
                        no_data_marker = -9999.0
                        rd_input = np.where(np.isfinite(dem_win), dem_win, no_data_marker).astype(np.float32)
                        dem_rd = rd.rdarray(rd_input, no_data=no_data_marker)
                        slope_win = rd.TerrainAttribute(dem_rd, "slope_degrees")
                    else:
                        dy, dx = np.gradient(dem_win)
                        slope_win = np.rad2deg(np.arctan(np.sqrt(dx * dx + dy * dy)))

                    # Remove padding to produce exact interior crop
                    y0 = pad if (ymax - ymin) > (2 * pad) else 0
                    x0 = pad if (xmax - xmin) > (2 * pad) else 0
                    y1 = (ymax - ymin) - pad if (ymax - ymin) > (2 * pad) else (ymax - ymin)
                    x1 = (xmax - xmin) - pad if (xmax - xmin) > (2 * pad) else (xmax - xmin)
                    dem_crop = dem_win[y0:y1, x0:x1]
                    slope_crop = slope_win[y0:y1, x0:x1]

                    # If crop window too small, fall back to minimal center pixel
                    if dem_crop.size == 0 or slope_crop.size == 0:
                        dem_crop = dem_win
                        slope_crop = slope_win

                    # Final sanitize for downstream consumers
                    dem_crop = dem_crop.astype(np.float32, copy=False)
                    slope_crop = slope_crop.astype(np.float32, copy=False)
                    self._debug_record('_read_cropped_dem_and_slope', 'done_rasterio', {
                        'win_shape': (int(dem_win.shape[0]), int(dem_win.shape[1])),
                        'crop_shape': (int(dem_crop.shape[0]), int(dem_crop.shape[1]))
                    })
                    return dem_crop, slope_crop
            except Exception as e:
                self._debug_record('_read_cropped_dem_and_slope', 'rasterio_error', {'error': str(e)})

        # Fallbacks: GDAL → PIL
        try:
            if GDAL_AVAILABLE:
                ds = gdal.Open(str(dem_file))
                dem_full = ds.ReadAsArray().astype(np.float32)
                gt = ds.GetGeoTransform()
                h_full, w_full = dem_full.shape
            else:
                with Image.open(str(dem_file)) as im:
                    dem_full = np.array(im).astype(np.float32)
                # Approximate GeoTransform for ~10 m pixels
                px_m = float(self.ned_resolution_m)
                deg_per_m_lat = 1.0 / 110540.0
                deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)) + 1e-9)
                xres = px_m * deg_per_m_lon
                yres = -px_m * deg_per_m_lat
                minx = lon - (dem_full.shape[1] * xres) / 2.0
                maxy = lat - (dem_full.shape[0] * yres) / 2.0
                gt = (minx, xres, 0.0, maxy, 0.0, yres)
                h_full, w_full = dem_full.shape

            # Center pixel and padded bounds
            px_c, py_c = self._latlon_to_pixel(gt, lat, lon)
            pad = int(max(1, pad_px))
            xmin = max(0, px_c - (pixels + pad))
            xmax = min(w_full, px_c + (pixels + pad))
            ymin = max(0, py_c - (pixels + pad))
            ymax = min(h_full, py_c + (pixels + pad))
            dem_win = dem_full[ymin:ymax, xmin:xmax]
            # Sanitize
            try:
                dem_win = np.where(dem_win <= -1e5, np.nan, dem_win)
            except Exception:
                pass
            if not np.isfinite(dem_win).any():
                dem_win = np.zeros_like(dem_win, dtype=np.float32)
            else:
                mean_val = float(np.nanmean(dem_win))
                dem_win = np.where(np.isfinite(dem_win), dem_win, mean_val)

            if RICHDEM_AVAILABLE:
                no_data_marker = -9999.0
                rd_input = np.where(np.isfinite(dem_win), dem_win, no_data_marker).astype(np.float32)
                dem_rd = rd.rdarray(rd_input, no_data=no_data_marker)
                slope_win = rd.TerrainAttribute(dem_rd, "slope_degrees")
            else:
                dy, dx = np.gradient(dem_win)
                slope_win = np.rad2deg(np.arctan(np.sqrt(dx * dx + dy * dy)))

            y0 = pad if (ymax - ymin) > (2 * pad) else 0
            x0 = pad if (xmax - xmin) > (2 * pad) else 0
            y1 = (ymax - ymin) - pad if (ymax - ymin) > (2 * pad) else (ymax - ymin)
            x1 = (xmax - xmin) - pad if (xmax - xmin) > (2 * pad) else (xmax - xmin)
            dem_crop = dem_win[y0:y1, x0:x1]
            slope_crop = slope_win[y0:y1, x0:x1]
            dem_crop = dem_crop.astype(np.float32, copy=False)
            slope_crop = slope_crop.astype(np.float32, copy=False)
            self._debug_record('_read_cropped_dem_and_slope', 'done_fallback', {
                'win_shape': (int(dem_win.shape[0]), int(dem_win.shape[1])),
                'crop_shape': (int(dem_crop.shape[0]), int(dem_crop.shape[1]))
            })
            return dem_crop, slope_crop
        except Exception as e:
            self._debug_record('_read_cropped_dem_and_slope', 'fallback_error', {'error': str(e)})
            # As absolute fallback, compute via full path
            attrs_full = self._load_dem_and_attributes(dem_file)
            if GDAL_AVAILABLE:
                ds = gdal.Open(str(dem_file))
                dem_full = ds.ReadAsArray().astype(np.float32)
                gt = ds.GetGeoTransform()
                h_full, w_full = dem_full.shape
            else:
                with Image.open(str(dem_file)) as im:
                    dem_full = np.array(im).astype(np.float32)
                h_full, w_full = dem_full.shape
            # Crop using existing helper
            ys, xs = self._crop_window(gt, w_full, h_full, lat, lon, half_side_m=half_side_m)
            return dem_full[ys, xs], attrs_full['slope_degrees'][ys, xs]

    def _download_elevation_data(self, region: List[float], out_path: str, lat: float, lon: float) -> bool:
        """
        Download a NED GeoTIFF for the bounding box. Validate with rasterio.
        - Try all URLs from leafmap; accept first valid GeoTIFF (>=64x64 and contains target)
        - Fallback to OpenTopography
        - Fallback to local .tif (skip .part)
        """
        self._debug_record('_download_elevation_data', 'start', {
            'region': list(map(float, region)),
            'target_lat': float(lat),
            'target_lon': float(lon),
            'out_path': str(out_path),
        })

        try:
            debug_attempts: List[Dict[str, object]] = []
            url_list = leafmap.download_ned(region, return_url=True)
            self._debug_record('_download_elevation_data', 'leafmap_urls', {
                'url_count': len(url_list),
                'urls': url_list[:3] if url_list else [],  # First 3 for brevity
            })

            if not url_list:
                warnings.warn("No NED tiles found for region")
                url_list = []

            for i, candidate_url in enumerate(url_list):
                try:
                    tmp_fp = str(Path(out_path).with_suffix('.tmp.tif'))
                    self._debug_record('_download_elevation_data', f'url_attempt_{i}', {'url': candidate_url})

                    # Download using leafmap; fallback to requests
                    try:
                        leafmap.download_file(candidate_url, tmp_fp, overwrite=True)
                        self._debug_record('_download_elevation_data', f'url_attempt_{i}', {'download_method': 'leafmap'})
                    except Exception as dl_err:
                        self._debug_record('_download_elevation_data', f'url_attempt_{i}', {'download_method': 'requests', 'error': str(dl_err)})
                        headers = {'User-Agent': 'LandslidePredictor/1.0 (+https://github.com/)'}
                        r = requests.get(candidate_url, timeout=180, headers=headers)
                        r.raise_for_status()
                        Path(tmp_fp).write_bytes(r.content)

                    # Validate
                    is_valid = False
                    w = h = 0
                    rc_ok = False
                    nodata_val = None
                    if RASTERIO_AVAILABLE:
                        try:
                            with rio.open(tmp_fp) as ds_v:
                                w, h = ds_v.width, ds_v.height
                                try:
                                    nodata_val = ds_v.nodata
                                except Exception:
                                    nodata_val = None
                                if w >= 64 and h >= 64:
                                    tr = ds_v.transform
                                    r, c = rio_rowcol(tr, lon, lat)
                                    rc_ok = (0 <= int(c) < w and 0 <= int(r) < h)
                                    if rc_ok:
                                        is_valid = True
                        except Exception as _v:
                            is_valid = False
                            self._debug_record('_download_elevation_data', f'url_attempt_{i}', {'rasterio_error': str(_v)})
                    else:
                        is_valid = True

                    if is_valid:
                        Path(out_path).write_bytes(Path(tmp_fp).read_bytes())
                        self._last_debug_download = {
                            'url': candidate_url,
                            'validated': True,
                            'shape': (int(h), int(w)),
                            'rowcol_in_bounds': rc_ok,
                            'region_used': list(map(float, region)),
                            'nodata': float(nodata_val) if nodata_val is not None else None,
                        }
                        self._debug_record('_download_elevation_data', f'url_success_{i}', {
                            'validated': True,
                            'shape': (int(h), int(w)),
                            'rowcol_in_bounds': rc_ok,
                            'nodata': float(nodata_val) if nodata_val is not None else None,
                        })
                        try:
                            Path(tmp_fp).unlink()
                        except Exception:
                            pass
                        return True
                    else:
                        debug_attempts.append({'url': candidate_url, 'validated': False, 'shape': (int(h), int(w)), 'rowcol_in_bounds': rc_ok})
                        self._debug_record('_download_elevation_data', f'url_reject_{i}', {
                            'reason': 'validation_failed',
                            'shape': (int(h), int(w)),
                            'rowcol_in_bounds': rc_ok,
                        })
                        try:
                            Path(tmp_fp).unlink()
                        except Exception:
                            pass
                except Exception as _dl_err:
                    warnings.warn(f"Leafmap URL failed validation: {_dl_err}")
                    debug_attempts.append({'url': candidate_url, 'error': str(_dl_err)})
                    self._debug_record('_download_elevation_data', f'url_error_{i}', {'error': str(_dl_err)})

        except Exception as e:
            warnings.warn(f"TNM URL list path failed: {e}")
            self._last_debug_download = {'error': str(e), 'region_used': list(map(float, region))}
            self._debug_record('_download_elevation_data', 'error', {'error': str(e)})

        # Try OpenTopography API (USGS NED 10m) with fallback to COP30
        self._debug_record('_download_elevation_data', 'opentopo_start', {'region': list(map(float, region))})
        try:
            min_lon, min_lat, max_lon, max_lat = region[0], region[1], region[2], region[3]
            api_key = os.environ.get("OPENTOPO_API_KEY")
            if api_key:
                ot_base = "https://portal.opentopography.org/API/globaldem"
                for demtype in ("USGSNED10m", "COP30"):
                    self._debug_record('_download_elevation_data', f'opentopo_attempt_{demtype}', {
                        'demtype': demtype,
                        'region': [min_lon, min_lat, max_lon, max_lat],
                    })
                    params = {
                        "demtype": demtype,
                        "south": f"{min_lat}",
                        "north": f"{max_lat}",
                        "west": f"{min_lon}",
                        "east": f"{max_lon}",
                        "API_Key": api_key,
                        "format": "GTiff"
                    }
                    resp = requests.get(ot_base, params=params, timeout=180)
                    if resp.status_code == 200 and resp.content:
                        Path(out_path).write_bytes(resp.content)
                        self._last_debug_download = {
                            'url': f'opentopography_{demtype}',
                            'validated': True,
                            'region_used': list(map(float, region)),
                        }
                        self._debug_record('_download_elevation_data', f'opentopo_success_{demtype}', {
                            'status_code': resp.status_code,
                            'content_length': len(resp.content),
                        })
                        return True
                    else:
                        warnings.warn(f"OpenTopography {demtype} failed: {resp.status_code} {resp.text[:120]}")
                        self._debug_record('_download_elevation_data', f'opentopo_fail_{demtype}', {
                            'status_code': resp.status_code,
                            'response_preview': resp.text[:120] if resp.text else '',
                        })
        except Exception as e:
            warnings.warn(f"OpenTopography path failed: {e}")
            self._debug_record('_download_elevation_data', 'opentopo_error', {'error': str(e)})

        # Local fallback: search repo data directories for a matching USGS tile
        self._debug_record('_download_elevation_data', 'local_fallback_start', {'region': list(map(float, region))})
        try:
            tile_n = math.ceil(lat)
            tile_w = abs(math.floor(lon))
            tile_str = f"n{tile_n}w{tile_w}"
            base_data = Path(__file__).parent.parent / 'data'
            candidates = []
            for d in [base_data, base_data / 'data']:
                try:
                    if d.is_dir():
                        for p in d.glob(f"**/*{tile_str}*.tif"):
                            # Only accept complete GeoTIFFs, skip partial temp files
                            if p.is_file() and p.suffix.lower() == '.tif' and not p.name.endswith('.part') and ("USGS_13_" in p.name or "USGS_" in p.name):
                                candidates.append(p)
                except Exception:
                    pass
            self._debug_record('_download_elevation_data', 'local_candidates', {
                'tile_str': tile_str,
                'candidate_count': len(candidates),
                'candidates': [p.name for p in candidates],
            })
            if candidates:
                # Prefer the most recent file
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                try:
                    Path(out_path).write_bytes(candidates[0].read_bytes())
                    self._last_debug_download = {
                        'url': 'local_fallback',
                        'validated': True,
                        'region_used': list(map(float, region)),
                        'local_file': candidates[0].name,
                        'local_path': str(candidates[0]),
                    }
                    self._debug_record('_download_elevation_data', 'local_success', {
                        'local_file': candidates[0].name,
                        'file_size': candidates[0].stat().st_size,
                    })
                    return True
                except Exception as _copy_err:
                    warnings.warn(f"Failed to copy local DEM tile {candidates[0].name}: {_copy_err}")
                    self._debug_record('_download_elevation_data', 'local_copy_error', {
                        'error': str(_copy_err),
                        'file': candidates[0].name,
                    })

        except Exception as e:
            warnings.warn(f"Local DEM fallback failed: {e}")
            self._debug_record('_download_elevation_data', 'local_error', {'error': str(e)})
        try:
            self._last_debug_download = {
                'attempts': debug_attempts,
                'region_used': list(map(float, region)),
                'final_fallback': 'failed',
            }
        except Exception:
            pass

        return False

    def _get_us_border_polygon(self, shapefile_path: str) -> Dict[str, Polygon]:
        """
        Load US border polygons from shapefile.
        
        Args:
            shapefile_path: Path to shapefile
            
        Returns:
            Dictionary mapping state names to polygons
        """
        sf = shapefile.Reader(shapefile_path)
        polys = {}
        for shape_rec in zip(sf.shapes(), sf.records()):
            poly = Polygon(shape_rec[0].points)
            state_name = shape_rec[1][5]  # attribute layout of US Census shapefile
            polys[state_name] = poly
        return polys

    def _compute_terrain_attributes(self, dem) -> Dict[str, np.ndarray]:
        """
        Compute terrain attributes from DEM using RichDEM or fallback.
        
        Args:
            dem: RichDEM array or numpy array of elevation data
            
        Returns:
            Dictionary containing terrain attributes
        """
        if RICHDEM_AVAILABLE:
            return {
                "slope_degrees": rd.TerrainAttribute(dem, "slope_degrees"),
                "aspect": rd.TerrainAttribute(dem, "aspect"),
                "planform_curvature": rd.TerrainAttribute(dem, "planform_curvature"),
                "profile_curvature": rd.TerrainAttribute(dem, "profile_curvature"),
            }
        else:
            # Fallback: simple gradient-based slope calculation
            if hasattr(dem, 'array'):
                elevation = dem.array
            else:
                elevation = dem if isinstance(dem, np.ndarray) else np.array(dem)
            
            # Calculate slope using numpy gradient (basic approximation)
            dy, dx = np.gradient(elevation)
            slope_rad = np.arctan(np.sqrt(dx*dx + dy*dy))
            slope_degrees = np.rad2deg(slope_rad)
            
            return {
                "slope_degrees": slope_degrees,
                "aspect": np.zeros_like(slope_degrees),  # Placeholder
                "planform_curvature": np.zeros_like(slope_degrees),  # Placeholder
                "profile_curvature": np.zeros_like(slope_degrees),  # Placeholder
            }

    def _load_dem_and_attributes(self, fp: str) -> Dict[str, np.ndarray]:
        """
        Load DEM and compute terrain attributes.
        
        Args:
            fp: Path to DEM file
            
        Returns:
            Dictionary of terrain attributes
        """
        self._debug_record('_load_dem_and_attributes', 'start', {'file_path': str(fp)})
        try:
            # Prefer rasterio to honor GDAL nodata; fallback to PIL
            arr: np.ndarray
            nodata_val: Optional[float] = None
            reader = 'unknown'
            if RASTERIO_AVAILABLE:
                with rio.open(fp) as ds:
                    arr = ds.read(1).astype(np.float32)
                    reader = 'rasterio'
                    try:
                        nodata_val = ds.nodata
                    except Exception:
                        nodata_val = None
            else:
                with Image.open(fp) as im:
                    arr = np.array(im).astype(np.float32)
                    reader = 'pil'
            self._debug_record('_load_dem_and_attributes', 'raw_read', {
                'reader': reader,
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
                'nodata': float(nodata_val) if nodata_val is not None else None,
                'min': float(np.nanmin(arr)) if arr.size else None,
                'max': float(np.nanmax(arr)) if arr.size else None,
                'finite_count': int(np.isfinite(arr).sum()),
                'total_pixels': int(arr.size),
            })
            if nodata_val is not None:
                arr = np.where(arr == nodata_val, np.nan, arr)
            # Also treat extreme negative sentinels as nodata
            try:
                arr = np.where(arr <= -1e5, np.nan, arr)
            except Exception:
                pass
            try:
                finite_ratio_pre = float(np.isfinite(arr).sum()) / float(arr.size)
            except Exception:
                finite_ratio_pre = None
            self._debug_record('_load_dem_and_attributes', 'after_mask', {
                'finite_ratio': finite_ratio_pre,
                'min': float(np.nanmin(arr)) if arr.size else None,
                'max': float(np.nanmax(arr)) if arr.size else None,
            })
            # Replace remaining NaNs with local mean to avoid holes
            if not np.isfinite(arr).any():
                arr = np.zeros_like(arr, dtype=np.float32)
            else:
                mean_val = float(np.nanmean(arr))
                arr = np.where(np.isfinite(arr), arr, mean_val)
            # Reject arrays that are mostly nodata to trigger retries upstream
            try:
                finite_ratio = float(np.isfinite(arr).sum()) / float(arr.size)
                if finite_ratio < 0.1:
                    raise RuntimeError("DEM content mostly nodata")
            except Exception:
                pass
            self._last_dem_read = {
                'file_path': str(fp),
                'reader': reader,
                'nodata': float(nodata_val) if nodata_val is not None else None,
                'finite_ratio_after_fill': float(np.isfinite(arr).sum()) / float(arr.size) if arr.size else None,
                'min_after_fill': float(np.nanmin(arr)) if arr.size else None,
                'max_after_fill': float(np.nanmax(arr)) if arr.size else None,
            }
            self._debug_record('_load_dem_and_attributes', 'after_fill', dict(self._last_dem_read))
            if RICHDEM_AVAILABLE:
                no_data_marker = -9999.0
                rd_input = np.where(np.isfinite(arr), arr, no_data_marker).astype(np.float32)
                imarray_rd = rd.rdarray(rd_input, no_data=no_data_marker)
                self._debug_record('_load_dem_and_attributes', 'richdem_ready', {
                    'no_data_marker': -9999.0,
                })
            else:
                imarray_rd = arr
            attrs = self._compute_terrain_attributes(imarray_rd)
            try:
                self._debug_record('_load_dem_and_attributes', 'attributes', {
                    'slope_min': float(np.nanmin(attrs.get('slope_degrees'))),
                    'slope_max': float(np.nanmax(attrs.get('slope_degrees'))),
                    'aspect_min': float(np.nanmin(attrs.get('aspect'))),
                    'aspect_max': float(np.nanmax(attrs.get('aspect'))),
                })
            except Exception:
                pass
            return attrs
        except Exception as e:
            self._debug_record('_load_dem_and_attributes', 'error', {'error': str(e)})
            raise RuntimeError(f"Failed to load DEM and compute attributes: {e}")

    def _offset(self, lat: float, lon: float, metres: float = 10_000) -> List[float]:
        """
        Calculate bounding box offset from point.
        
        Args:
            lat: Latitude
            lon: Longitude
            metres: Offset distance in meters
            
        Returns:
            Bounding box [min_lon, min_lat, max_lon, max_lat]
        """
        R = 6_378_137.0
        d_lat = metres / R
        d_lon = metres / (R * math.cos(math.radians(lat)))
        return [
            lon - math.degrees(d_lon),
            lat - math.degrees(d_lat),
            lon + math.degrees(d_lon),
            lat + math.degrees(d_lat),
        ]

    def _latlon_to_pixel(self, gt: Tuple, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert lat/lon to pixel coordinates.
        
        Args:
            gt: GeoTransform tuple
            lat: Latitude
            lon: Longitude
            
        Returns:
            Pixel coordinates (x, y)
        """
        minx, xres, _, maxy, _, yres = gt
        px = int((lon - minx) / xres)
        py = int((maxy - lat) / -yres)
        return px, py

    def get_terrain_features_for_point(self, lat: float, lon: float,
                                     output_dir: str = "temp_dem") -> Dict[str, float]:
        """
        Get terrain features for a single point.
        
        Args:
            lat: Latitude
            lon: Longitude
            output_dir: Directory to save temporary files
            
        Returns:
            Dictionary of terrain features
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Try progressively larger regions to ensure the point is within bounds
        region_sizes_m = [2000, 10000, 20000]

        last_error: Optional[Exception] = None
        self._debug_record('get_terrain_features_for_point', 'start', {
            'lat': float(lat), 'lon': float(lon), 'output_dir': str(output_dir),
            'region_sizes_m': list(map(int, region_sizes_m)),
        })
        for metres in region_sizes_m:
            dem_file = Path(output_dir) / f"dem_{lat:.5f}_{lon:.5f}_{metres}.tif"
            try:
                region = self._offset(lat, lon, metres=metres)
                self._debug_record('get_terrain_features_for_point', 'attempt', {
                    'metres': int(metres), 'dem_file': dem_file.name, 'region': list(map(float, region)),
                })
                if not self._download_elevation_data(region, str(dem_file), lat, lon):
                    last_error = RuntimeError(f"Failed to download DEM for point ({lat}, {lon}) at {metres} m window")
                    self._debug_record('get_terrain_features_for_point', 'download_failed', {
                        'metres': int(metres), 'dem_file': dem_file.name,
                    })
                    continue

                # Compute attributes using a small padded window around the pixel to
                # keep values identical to full-raster computation at the center.
                # Use a modest window (e.g., 50 m) to bound memory.
                window_half_m = max(self.ned_resolution_m * 5, 50)
                dem_c, _slope_c = self._read_cropped_dem_and_slope(str(dem_file), lat, lon,
                                                                   half_side_m=int(window_half_m), pad_px=2)
                # Derive all attributes on the same window so center value matches
                if RICHDEM_AVAILABLE:
                    no_data_marker = -9999.0
                    rd_input = np.where(np.isfinite(dem_c), dem_c, no_data_marker).astype(np.float32)
                    dem_rd = rd.rdarray(rd_input, no_data=no_data_marker)
                    attrs_w = {
                        "slope_degrees": rd.TerrainAttribute(dem_rd, "slope_degrees"),
                        "aspect": rd.TerrainAttribute(dem_rd, "aspect"),
                        "planform_curvature": rd.TerrainAttribute(dem_rd, "planform_curvature"),
                        "profile_curvature": rd.TerrainAttribute(dem_rd, "profile_curvature"),
                    }
                else:
                    # Fallback: compute minimal set matching existing behavior
                    dy, dx = np.gradient(dem_c)
                    slope_rad = np.arctan(np.sqrt(dx * dx + dy * dy))
                    slope_degrees = np.rad2deg(slope_rad)
                    zero_like = np.zeros_like(slope_degrees)
                    attrs_w = {
                        "slope_degrees": slope_degrees,
                        "aspect": zero_like,
                        "planform_curvature": zero_like,
                        "profile_curvature": zero_like,
                    }

                # Get pixel coordinates and image shape via available backend
                gt = None
                w = h = None
                px = py = None
                if GDAL_AVAILABLE:
                    ds = gdal.Open(str(dem_file))
                    gt = ds.GetGeoTransform()
                    h, w = ds.ReadAsArray().shape
                    px, py = self._latlon_to_pixel(gt, lat, lon)
                elif RASTERIO_AVAILABLE:
                    with rio.open(str(dem_file)) as ds_r:
                        tr = ds_r.transform
                        w, h = ds_r.width, ds_r.height
                        # row, col from lon/lat
                        r, c = rio_rowcol(tr, lon, lat)
                        px, py = int(c), int(r)
                else:
                    # Fallback: approximate geotransform assuming ~10 m pixels
                    img = Image.open(str(dem_file))
                    arr = np.array(img)
                    h, w = arr.shape
                    px_m = float(self.ned_resolution_m)
                    deg_per_m_lat = 1.0 / 110540.0
                    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)) + 1e-9)
                    xres = px_m * deg_per_m_lon
                    yres = -px_m * deg_per_m_lat
                    minx = lon - (w * xres) / 2.0
                    maxy = lat - (h * yres) / 2.0
                    gt = (minx, xres, 0.0, maxy, 0.0, yres)
                    px, py = self._latlon_to_pixel(gt, lat, lon)

                # Clamp to valid bounds to handle edge/rounding cases
                clamped_px = min(max(px, 0), w - 1)
                clamped_py = min(max(py, 0), h - 1)
                self._debug_record('get_terrain_features_for_point', 'pixel', {
                    'metres': int(metres), 'dem_shape': (int(h), int(w)),
                    'px_py_raw': (int(px), int(py)), 'px_py_clamped': (int(clamped_px), int(clamped_py)),
                })

                # Extract values at clamped pixel using the window-based attributes.
                # Align indices to center of window crop
                wy, wx = attrs_w['slope_degrees'].shape
                cy, cx = int(wy // 2), int(wx // 2)
                try:
                    features = {k: float(v[cy, cx]) for k, v in attrs_w.items()}
                except Exception:
                    # Fallback to clamped indices in case of odd windowing near tile edges
                    cy = min(max(0, cy), wy - 1)
                    cx = min(max(0, cx), wx - 1)
                    features = {k: float(v[cy, cx]) for k, v in attrs_w.items()}
                # Store debug details for callers (e.g., Streamlit UI)
                try:
                    self._last_debug_usgs = {
                        'dem_file': str(dem_file.name),
                        'dem_shape': (int(h), int(w)),
                        'pixel_at': (int(clamped_py), int(clamped_px)),
                        'slope_min': float(np.nanmin(attrs_w.get('slope_degrees'))),
                        'slope_max': float(np.nanmax(attrs_w.get('slope_degrees'))),
                        'region_m': int(metres),
                        'value_at_pixel': {k: float(features.get(k, float('nan'))) for k in features.keys()},
                    }
                except Exception:
                    self._last_debug_usgs = {'dem_file': str(dem_file.name)}
                self._debug_record('get_terrain_features_for_point', 'success', dict(self._last_debug_usgs))

                # Clean up
                try:
                    ds = None
                except Exception:
                    pass
                return features

            except Exception as e:
                last_error = e
                # Preserve DEM file for potential reuse by 3D renderer
                self._debug_record('get_terrain_features_for_point', 'attempt_error', {
                    'metres': int(metres), 'dem_file': dem_file.name, 'error': str(e),
                })
                continue

        # As a final fallback, return neutral/default terrain values instead of failing
        warnings.warn(f"DEM lookup failed for ({lat}, {lon}); using fallback terrain values. Last error: {last_error}")
        return {
            "slope_degrees": float('nan'),
            "aspect": float('nan'),
            "planform_curvature": float('nan'),
            "profile_curvature": float('nan'),
        }

    def collect_terrain_data_batch(self, coordinates: List[Tuple[float, float]], 
                                 output_dir: str = "temp_dem") -> pd.DataFrame:
        """
        Collect terrain data for multiple coordinates.
        
        Args:
            coordinates: List of (lat, lon) tuples
            output_dir: Directory for temporary files
            
        Returns:
            DataFrame with terrain features for each coordinate
        """
        results = []
        
        for i, (lat, lon) in enumerate(coordinates):
            try:
                print(f"Processing coordinate {i+1}/{len(coordinates)}: ({lat:.5f}, {lon:.5f})")
                features = self.get_terrain_features_for_point(lat, lon, output_dir)
                features['latitude'] = lat
                features['longitude'] = lon
                results.append(features)
            except Exception as e:
                print(f"Failed to process coordinate ({lat}, {lon}): {e}")
                # Add row with NaN values
                results.append({
                    'latitude': lat,
                    'longitude': lon,
                    'slope_degrees': np.nan,
                    'aspect': np.nan,
                    'planform_curvature': np.nan,
                    'profile_curvature': np.nan
                })
        
        return pd.DataFrame(results)

    def plot_terrain_maps(self, lat: float, lon: float, 
                         output_dir: str = "temp_dem", 
                         save_plots: bool = True) -> None:
        """
        Create and optionally save terrain visualization plots.
        
        Args:
            lat: Latitude
            lon: Longitude
            output_dir: Directory for temporary files
            save_plots: Whether to save plots to files
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Download DEM data
        region = self._offset(lat, lon)
        dem_file = Path(output_dir) / f"dem_plot_{lat:.5f}_{lon:.5f}.tif"
        
        if not self._download_elevation_data(region, str(dem_file), lat, lon):
            print(f"Failed to download DEM for plotting at ({lat}, {lon})")
            return
        
        try:
            # Plot DEM
            self._plot_dem(lat, lon, str(dem_file), save_plots, output_dir)
            
            # Plot slope
            self._plot_slope_map(lat, lon, str(dem_file), save_plots, output_dir)
            
        finally:
            # Clean up
            if dem_file.exists():
                try:
                    dem_file.unlink()
                except Exception:
                    pass

    def _plot_dem(self, lat: float, lon: float, dem_file: str, 
                 save_plots: bool, output_dir: str) -> None:
        """Plot DEM elevation data."""
        # Read only a small crop around the point to reduce memory
        try:
            dem_c, _slope_c = self._read_cropped_dem_and_slope(dem_file, lat, lon, half_side_m=200, pad_px=2)
            cropped = dem_c
        except Exception as e:
            raise RuntimeError(f"Failed to read DEM for plotting: {e}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cropped, cmap='terrain', extent=[-200, 200, -200, 200], origin="upper")
        fig.colorbar(im, ax=ax, label='Elevation (m)')
        
        # Add center marker
        ax.scatter(0, 0, marker="x", s=100, c="red", linewidths=2, label="Location")
        
        # Format plot
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_aspect("equal")
        ax.set_xlabel("m East/West")
        ax.set_ylabel("m North/South")
        ax.set_title(f"DEM (400 m × 400 m) – lat {lat:.5f}, lon {lon:.5f}")
        ax.legend(loc="upper right")
        
        if save_plots:
            plot_file = Path(output_dir) / f"dem_plot_{lat:.5f}_{lon:.5f}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"DEM plot saved to {plot_file}")
        
        plt.show()

    def _plot_slope_map(self, lat: float, lon: float, dem_file: str, 
                       save_plots: bool, output_dir: str) -> None:
        """Plot slope data."""
        # Compute slope on a padded crop then display the interior crop
        try:
            _dem_c, slope_c = self._read_cropped_dem_and_slope(dem_file, lat, lon, half_side_m=200, pad_px=2)
            cropped = slope_c
        except Exception as e:
            raise RuntimeError(f"Failed to read DEM for slope plotting: {e}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cropped, cmap='plasma', extent=[-200, 200, -200, 200], origin="upper")
        fig.colorbar(im, ax=ax, label='Slope (°)')
        
        # Add center marker
        ax.scatter(0, 0, marker="x", s=100, c="red", linewidths=2, label="Location")
        
        # Format plot
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_aspect("equal")
        ax.set_xlabel("m East/West")
        ax.set_ylabel("m North/South")
        ax.set_title(f"Slope Angle (400 m × 400 m) – lat {lat:.5f}, lon {lon:.5f}")
        ax.legend(loc="upper right")
        
        if save_plots:
            plot_file = Path(output_dir) / f"slope_plot_{lat:.5f}_{lon:.5f}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Slope plot saved to {plot_file}")
        
        plt.show()

    def plot_terrain_3d(self, lat: float, lon: float,
                        output_dir: str = "temp_dem",
                        half_side_m: int = 200,
                        save_plots: bool = False):
        """
        Build a 3D topographic visualization using DEM elevation as surface (Z)
        and slope (degrees) as face color. A red point marks the target location.

        Returns a matplotlib Figure for embedding (e.g., in Streamlit).
        """
        Path(output_dir).mkdir(exist_ok=True)

        # Download DEM data around the point (small window for performance)
        region = self._offset(lat, lon, metres=max(2_000, half_side_m * 4))
        dem_file = Path(output_dir) / f"dem_3d_{lat:.5f}_{lon:.5f}.tif"
        if not self._download_elevation_data(region, str(dem_file), lat, lon):
            raise RuntimeError(f"Failed to download DEM for 3D plot at ({lat}, {lon})")

        try:
            # Read a padded crop and compute slope on it to reduce memory
            dem_c, slope_c = self._read_cropped_dem_and_slope(str(dem_file), lat, lon,
                                                              half_side_m=half_side_m, pad_px=2)

            # If crop too small/flat, try a larger crop once
            try:
                size_y_chk, size_x_chk = dem_c.shape
                if size_x_chk < 3 or size_y_chk < 3 or (
                    (np.nanstd(dem_c) < 1e-6) and (np.nanstd(slope_c) < 1e-6)
                ):
                    # Re-read with a larger crop to ensure meaningful mesh
                    larger_half = int(max(half_side_m * 2, 800))
                    dem_c, slope_c = self._read_cropped_dem_and_slope(str(dem_file), lat, lon,
                                                                      half_side_m=larger_half, pad_px=2)
            except Exception:
                pass

            # Downsample to keep mesh reasonable (<= 150 x 150)
            size_y, size_x = dem_c.shape
            target = 150
            stride = int(max(1, np.ceil(max(size_x, size_y) / target)))
            if stride > 1:
                dem_c = dem_c[::stride, ::stride]
                slope_c = slope_c[::stride, ::stride]

            # Create X, Y grids in meters for visualization extents
            size_y, size_x = dem_c.shape
            extent_m = half_side_m
            x_lin = np.linspace(-extent_m, extent_m, size_x)
            y_lin = np.linspace(-extent_m, extent_m, size_y)
            X, Y = np.meshgrid(x_lin, y_lin)

            # Normalize slope for colormap
            slope_norm = (slope_c - np.nanmin(slope_c)) / (np.nanmax(slope_c) - np.nanmin(slope_c) + 1e-9)
            cmap = plt.get_cmap('plasma')
            face_colors = cmap(slope_norm)

            # Center elevation for the red marker
            center_z = float(np.nanmean(dem_c))

            # Build figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, dem_c, facecolors=face_colors, rstride=1, cstride=1, linewidth=0, antialiased=False)
            ax.scatter([0], [0], [center_z], c='red', s=50, depthshade=False)
            ax.set_xlabel('m East/West')
            ax.set_ylabel('m North/South')
            ax.set_zlabel('Elevation (m)')
            ax.set_title(f"3D Topography (colored by slope) – lat {lat:.5f}, lon {lon:.5f}")

            if save_plots:
                out_png = Path(output_dir) / f"topography_3d_{lat:.5f}_{lon:.5f}.png"
                fig.savefig(out_png, dpi=200, bbox_inches='tight')

            return fig
        finally:
            if 'ds' in locals() and ds is not None:
                ds = None
            if dem_file.exists():
                try:
                    dem_file.unlink()
                except Exception:
                    pass

    def build_interactive_3d(self, lat: float, lon: float,
                              output_dir: str = "temp_dem",
                              half_side_m: int = 200) -> Tuple[go.Figure, str]:
        """
        Build an interactive 3D topographic Plotly figure (pan/zoom/rotate).
        Strategy:
        - Reuse existing DEM from slope if available (smallest window)
        - Otherwise download a small window (~2 km)
        - Crop first, then compute slope on the cropped window only
        - If crop is invalid/flat, retry once with a larger window, then fail
        Returns (figure, resolution_label).
        """
        Path(output_dir).mkdir(exist_ok=True)
        self._debug_record('build_interactive_3d', 'start', {
            'lat': float(lat), 'lon': float(lon), 'output_dir': str(output_dir), 'half_side_m': int(half_side_m)
        })

        # Helper: render from a DEM file path and a crop size
        def _render_from_path(path: Path, crop_half_m: int) -> Tuple[go.Figure, str]:
            ds = None
            try:
                # Prefer rasterio for correct georeferencing; then GDAL; finally PIL
                if RASTERIO_AVAILABLE:
                    with rio.open(str(path)) as ds_r:
                        dem = ds_r.read(1).astype(np.float32)
                        nodata_val = None
                        try:
                            nodata_val = ds_r.nodata
                            if nodata_val is not None:
                                dem = np.where(dem == nodata_val, np.nan, dem)
                        except Exception:
                            pass
                        # Apply scale and offset if present
                        try:
                            if getattr(ds_r, 'scales', None) and ds_r.scales[0] not in (None, 1.0):
                                dem = dem * float(ds_r.scales[0])
                            if getattr(ds_r, 'offsets', None) and ds_r.offsets[0] not in (None, 0.0):
                                dem = dem + float(ds_r.offsets[0])
                        except Exception:
                            pass
                        dem = np.where(dem <= -1e5, np.nan, dem)
                        tr = ds_r.transform
                        gt = (tr.c, tr.a, tr.b, tr.f, tr.d, tr.e)
                        self._debug_record('build_interactive_3d', 'read_rasterio', {
                            'path': path.name,
                            'shape': (int(dem.shape[0]), int(dem.shape[1])),
                            'nodata': float(nodata_val) if nodata_val is not None else None,
                            'min': float(np.nanmin(dem)) if dem.size else None,
                            'max': float(np.nanmax(dem)) if dem.size else None,
                        })
                elif GDAL_AVAILABLE:
                    ds = gdal.Open(str(path))
                    band = ds.GetRasterBand(1)
                    dem = band.ReadAsArray().astype(np.float32)
                    try:
                        nodata_val = band.GetNoDataValue()
                        if nodata_val is not None:
                            dem = np.where(dem == nodata_val, np.nan, dem)
                    except Exception:
                        pass
                    dem = np.where(dem <= -1e5, np.nan, dem)
                    gt = ds.GetGeoTransform()
                    self._debug_record('build_interactive_3d', 'read_gdal', {
                        'path': path.name,
                        'shape': (int(dem.shape[0]), int(dem.shape[1])),
                        'nodata': float(nodata_val) if 'nodata_val' in locals() and nodata_val is not None else None,
                        'min': float(np.nanmin(dem)) if dem.size else None,
                        'max': float(np.nanmax(dem)) if dem.size else None,
                    })
                else:
                    img = Image.open(str(path))
                    dem = np.array(img).astype(np.float32)
                    dem = np.where(dem <= -1e5, np.nan, dem)
                    # Approx geotransform with ~10 m pixels
                    px_m = 10.0
                    deg_per_m_lat = 1.0 / 110540.0
                    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)) + 1e-9)
                    xres = px_m * deg_per_m_lon
                    yres = -px_m * deg_per_m_lat
                    minx = lon - (dem.shape[1] * xres) / 2.0
                    maxy = lat - (dem.shape[0] * yres) / 2.0
                    gt = (minx, xres, 0.0, maxy, 0.0, yres)
                    self._debug_record('build_interactive_3d', 'read_pil', {
                        'path': path.name,
                        'shape': (int(dem.shape[0]), int(dem.shape[1])),
                        'min': float(np.nanmin(dem)) if dem.size else None,
                        'max': float(np.nanmax(dem)) if dem.size else None,
                    })

                h, w = dem.shape
                ys, xs = self._crop_window(gt, w, h, lat, lon, half_side_m=crop_half_m)
                dem_c = dem[ys, xs]

                # Compute slope on the cropped window only
                if RICHDEM_AVAILABLE:
                    dem_rd = rd.rdarray(dem_c.astype(np.float32), no_data=-9999)
                    slope_c = rd.TerrainAttribute(dem_rd, "slope_degrees")
                else:
                    dy, dx = np.gradient(dem_c)
                    slope_c = np.rad2deg(np.arctan(np.sqrt(dx*dx + dy*dy)))

                # Validate and sanitize content instead of failing hard
                if not np.isfinite(dem_c).any():
                    # All values invalid; render a flat plane with tiny noise so it still displays
                    dem_c = np.zeros_like(dem_c, dtype=float)
                else:
                    # Replace NaNs with the local mean to keep surface contiguous
                    mean_val = float(np.nanmean(dem_c))
                    dem_c = np.where(np.isfinite(dem_c), dem_c, mean_val)
                # Sanitize slope colors too
                if slope_c is None or (not np.isfinite(slope_c).any()):
                    slope_c = np.zeros_like(dem_c, dtype=float)
                else:
                    slope_mean = float(np.nanmean(slope_c))
                    slope_c = np.where(np.isfinite(slope_c), slope_c, slope_mean)
                # If essentially flat, add a tiny perturbation to avoid degenerate rendering
                try:
                    if (np.nanmax(dem_c) - np.nanmin(dem_c)) < 1e-3 or (np.nanstd(dem_c) < 1e-3):
                        rng = np.random.default_rng(0)
                        dem_c = dem_c + 1e-3 * rng.standard_normal(dem_c.shape)
                except Exception:
                    pass

                # Downsample
                size_y, size_x = dem_c.shape
                target = 200
                stride = int(max(1, np.ceil(max(size_x, size_y) / target)))
                if stride > 1:
                    dem_c = dem_c[::stride, ::stride]
                    slope_c = slope_c[::stride, ::stride]
                size_y, size_x = dem_c.shape
                # If the cropped window is too small to render meaningfully, trigger a larger crop retry
                if size_x < 3 or size_y < 3:
                    raise ValueError("DEM crop too small for 3D rendering")
                self._debug_record('build_interactive_3d', 'crop_downsample', {
                    'path': path.name,
                    'crop_half_m': int(crop_half_m),
                    'crop_shape_after_stride': (int(size_y), int(size_x)),
                    'stride': int(stride),
                    'dem_min': float(np.nanmin(dem_c)),
                    'dem_max': float(np.nanmax(dem_c)),
                    'slope_min': float(np.nanmin(slope_c)),
                    'slope_max': float(np.nanmax(slope_c)),
                })

                # Sanitize for Plotly and build X/Y grids directly in meters
                size_y, size_x = dem_c.shape
                # Ensure plain float arrays and no NaNs for Plotly
                try:
                    dem_c = dem_c.astype(float)
                    slope_c = slope_c.astype(float)
                except Exception:
                    pass
                try:
                    # Replace any residual NaNs with local means
                    if not np.isfinite(dem_c).all():
                        dem_mean = float(np.nanmean(dem_c)) if np.isfinite(dem_c).any() else 0.0
                        dem_c = np.where(np.isfinite(dem_c), dem_c, dem_mean)
                    if not np.isfinite(slope_c).all():
                        slope_mean = float(np.nanmean(slope_c)) if np.isfinite(slope_c).any() else 0.0
                        slope_c = np.where(np.isfinite(slope_c), slope_c, slope_mean)
                except Exception:
                    pass
                x_lin = np.linspace(-crop_half_m, crop_half_m, size_x)
                y_lin = np.linspace(-crop_half_m, crop_half_m, size_y)
                X, Y = np.meshgrid(x_lin, y_lin)

                # Resolution label
                try:
                    m_per_deg_lat = 110540.0
                    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
                    px_lat_m = abs(gt[5]) * m_per_deg_lat if ds is not None else 10.0
                    px_lon_m = abs(gt[1]) * m_per_deg_lon if ds is not None else 10.0
                    px_m = (px_lat_m + px_lon_m) / 2.0
                    res_label = "10 m (1/3 arc-second)" if px_m <= 15.0 else "30 m (1 arc-second)"
                except Exception:
                    res_label = "Unknown resolution"

                # Center elevation
                px_c, py_c = self._latlon_to_pixel(gt, lat, lon)
                if 0 <= px_c < w and 0 <= py_c < h:
                    center_z = float(dem[int(py_c), int(px_c)])
                else:
                    center_z = float(np.nanmean(dem_c))

                # Clamp color range and ensure surface draws even with NaNs
                surface = go.Surface(
                    x=X,
                    y=Y,
                    z=dem_c,
                    surfacecolor=slope_c,
                    colorscale='Plasma',
                    colorbar=dict(title='Slope (°)'),
                    showscale=True,
                    opacity=1.0,
                    connectgaps=True,
                    cmin=float(np.nanmin(slope_c)),
                    cmax=float(np.nanmax(slope_c)),
                    contours=dict(
                        z=dict(show=True, usecolormap=False, color='rgba(0,0,0,0.35)', width=1)
                    ),
                )
                # Place marker at center of cropped window coordinates (0,0) in meter axes; lift it slightly above surface
                marker = go.Scatter3d(x=[0], y=[0], z=[center_z + 0.5], mode='markers', marker=dict(size=6, color='red'), name='Target')
                fig = go.Figure(data=[surface, marker])
                # Explicit axis ranges and aspect to avoid collapsed view
                zmin = float(np.nanmin(dem_c))
                zmax = float(np.nanmax(dem_c))
                scene_cfg = dict(
                    xaxis=dict(title='m East/West', range=[-crop_half_m, crop_half_m], visible=True, showgrid=True, zeroline=False, showspikes=False),
                    yaxis=dict(title='m North/South', range=[-crop_half_m, crop_half_m], visible=True, showgrid=True, zeroline=False, showspikes=False),
                    zaxis=dict(title='Elevation (m)', range=[zmin, zmax], visible=True, showgrid=True, zeroline=False, showspikes=False),
                    aspectmode='data',
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=30),
                    title=f"Interactive 3D Topography – lat {lat:.5f}, lon {lon:.5f}",
                    scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
                    scene=scene_cfg,
                    uirevision=True,
                    template=None,
                )
                # Save an HTML snapshot for external inspection
                html_path = None
                try:
                    html_path = Path(output_dir) / f"interactive_3d_{lat:.5f}_{lon:.5f}.html"
                    # Use dark theme look with white axes/text
                    fig_dark = fig.to_dict()
                    # Update scene and font for HTML
                    fig.update_layout(
                        paper_bgcolor='black',
                        plot_bgcolor='black',
                        font=dict(color='white', family='Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, Helvetica Neue, Arial, sans-serif'),
                        margin=dict(l=10, r=10, t=40, b=10),
                        title=dict(pad=dict(t=6, b=6), x=0.02, xanchor='left')
                    )
                    html_core = fig.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%', default_height='560px', config={'displayModeBar': False})
                    html_wrapped = (
                        "<style>\n"
                        "html,body{margin:0;padding:0;background:#000;}\n"
                        ".plotly,.js-plotly-plot,.plot-container{margin:0!important;padding:0!important;}\n"
                        ".modebar{display:none!important;}\n"
                        "</style>\n"
                        f"<div style='width:100%;height:100%;'>{html_core}</div>"
                    )
                    Path(html_path).write_text(html_wrapped, encoding='utf-8')
                except Exception:
                    html_path = None
                # Store debug info for callers
                try:
                    self._last_debug_3d = {
                        'dem_path': str(path.name),
                        'crop_half_m': int(crop_half_m),
                        'dem_shape': (int(h), int(w)),
                        'crop_shape': (int(size_y), int(size_x)),
                        'stride': int(stride),
                        'dem_min': float(np.nanmin(dem_c)),
                        'dem_max': float(np.nanmax(dem_c)),
                        'slope_min': float(np.nanmin(slope_c)),
                        'slope_max': float(np.nanmax(slope_c)),
                        'resolution_label': res_label,
                        'surface_points': int(size_x * size_y),
                        'nan_counts': {
                            'dem': int(np.isnan(dem_c).sum()),
                            'slope': int(np.isnan(slope_c).sum()),
                        },
                        'scene': scene_cfg,
                        'camera_eye': {'x': 1.6, 'y': 1.6, 'z': 0.8},
                        'html_path': str(html_path) if html_path else None,
                    }
                except Exception:
                    self._last_debug_3d = {'dem_path': str(path.name)}
                return fig, res_label
            finally:
                if 'ds' in locals() and ds is not None:
                    ds = None

        # Pick a reuse DEM if available (smallest window first)
        reuse = sorted(Path(output_dir).glob(f"dem_{lat:.5f}_{lon:.5f}_*.tif"),
                       key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 1_000_000)

        # Try reuse path → then small fresh download → then reuse with larger crop once
        base_half = half_side_m
        # Align default DEM window with slope feature first attempt (10 km)
        base_window_m = max(10_000, half_side_m * 4)
        fresh_path = Path(output_dir) / f"dem_3d_plotly_{lat:.5f}_{lon:.5f}_{base_window_m}.tif"

        # 1) Reuse (if present)
        if reuse:
            try:
                # Record reuse as a download debug entry
                self._last_debug_download = {'url': 'reuse', 'file': reuse[0].name, 'path': str(reuse[0])}
                self._debug_record('build_interactive_3d', 'reuse', {
                    'reuse_file': reuse[0].name,
                    'path': str(reuse[0]),
                })
                return _render_from_path(reuse[0], base_half)
            except Exception:
                pass

        # 2) Fresh download with robust window (align with slope feature)
        region = self._offset(lat, lon, metres=base_window_m)
        if not fresh_path.exists():
            if not self._download_elevation_data(region, str(fresh_path), lat, lon):
                raise RuntimeError(f"Failed to download DEM for interactive 3D at ({lat}, {lon})")
        else:
            # Even if exists, record as reuse-equivalent for transparency
            self._debug_record('build_interactive_3d', 'fresh_exists', {'path': str(fresh_path.name)})

        try:
            return _render_from_path(fresh_path, base_half)
        except Exception:
            # 3) Try once more with larger crop on the same file
            return _render_from_path(fresh_path, base_half * 2)

    def _crop_window(self, gt: Tuple, width: int, height: int, 
                    lat: float, lon: float, half_side_m: int = 200) -> Tuple[slice, slice]:
        """
        Calculate crop window around point.
        
        Args:
            gt: GeoTransform tuple
            width: Image width
            height: Image height
            lat: Latitude
            lon: Longitude
            half_side_m: Half side length in meters
            
        Returns:
            Tuple of slice objects for cropping
        """
        px_c, py_c = self._latlon_to_pixel(gt, lat, lon)
        pixels = int(half_side_m / self.ned_resolution_m)
        xmin, xmax = max(0, px_c - pixels), min(width, px_c + pixels)
        ymin, ymax = max(0, py_c - pixels), min(height, py_c + pixels)
        
        if xmax - xmin < 2 or ymax - ymin < 2:
            raise ValueError("Crop window too small - choose a coordinate further from tile edge.")
        
        return slice(ymin, ymax), slice(xmin, xmax)


def collect_slope_data_for_dataset(df: pd.DataFrame, 
                                 lat_col: str = 'Latitude',
                                 lon_col: str = 'Longitude',
                                 output_dir: str = "slope_data") -> pd.DataFrame:
    """
    Collect slope data for all coordinates in a dataset.
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        output_dir: Directory for output files
        
    Returns:
        DataFrame with original data plus terrain features
    """
    # Initialize collector
    collector = SlopeDataCollector()
    
    # Get coordinates
    coordinates = list(zip(df[lat_col], df[lon_col]))
    
    # Collect terrain data
    terrain_df = collector.collect_terrain_data_batch(coordinates, output_dir)
    
    # Merge with original data
    result_df = df.copy()
    terrain_cols = ['slope_degrees', 'aspect', 'planform_curvature', 'profile_curvature']
    
    for col in terrain_cols:
        if col in terrain_df.columns:
            result_df[f'terrain_{col}'] = terrain_df[col]
    
    return result_df


if __name__ == "__main__":
    # Example usage
    collector = SlopeDataCollector()
    
    # Test with a single point
    lat, lon = 37.54189, -120.96683
    try:
        features = collector.get_terrain_features_for_point(lat, lon)
        print(f"Terrain features for ({lat}, {lon}):")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"Error collecting terrain data: {e}")
    
    # Test batch collection
    coordinates = [(37.54189, -120.96683), (37.9014, -121.1895)]
    try:
        terrain_df = collector.collect_terrain_data_batch(coordinates)
        print(f"\nBatch collection results:")
        print(terrain_df)
    except Exception as e:
        print(f"Error in batch collection: {e}") 