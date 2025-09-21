"""
Slope Data Collector Module
Collects terrain and slope data using NED DEM tiles.
"""

import math
import random
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

    def _download_elevation_data(self, region: List[float], out_path: str) -> bool:
        """
        Download a single NED GeoTIFF for the bounding box.
        
        Args:
            region: Bounding box [min_lon, min_lat, max_lon, max_lat]
            out_path: Output file path
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            url = leafmap.download_ned(region, return_url=True)
            if not url:
                warnings.warn("No NED data found for region.")
                return False
                
            r = requests.get(url[0], timeout=60)
            r.raise_for_status()
            Path(out_path).write_bytes(r.content)
            return True
        except Exception as e:
            warnings.warn(f"Failed to download elevation data: {e}")
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
        try:
            im = Image.open(fp)
            imarray = np.array(im)
            if RICHDEM_AVAILABLE:
                imarray_rd = rd.rdarray(imarray, no_data=-9999)
            else:
                imarray_rd = imarray
            attrs = self._compute_terrain_attributes(imarray_rd)
            return attrs
        except Exception as e:
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
        region_sizes_m = [10_000, 50_000, 100_000]

        last_error: Optional[Exception] = None
        for metres in region_sizes_m:
            dem_file = Path(output_dir) / f"dem_{lat:.5f}_{lon:.5f}_{metres}.tif"
            try:
                region = self._offset(lat, lon, metres=metres)
                if not self._download_elevation_data(region, str(dem_file)):
                    last_error = RuntimeError(f"Failed to download DEM for point ({lat}, {lon}) at {metres} m window")
                    continue

                # Load DEM and compute attributes
                attrs = self._load_dem_and_attributes(str(dem_file))

                # Get pixel coordinates
                ds = gdal.Open(str(dem_file))
                gt = ds.GetGeoTransform()
                h, w = ds.ReadAsArray().shape
                px, py = self._latlon_to_pixel(gt, lat, lon)

                # If slightly out-of-bounds due to rounding, clamp to bounds
                clamped_px = min(max(px, 0), w - 1)
                clamped_py = min(max(py, 0), h - 1)

                if not (0 <= px < w and 0 <= py < h):
                    # Retry with next larger region if available
                    last_error = ValueError(f"Point ({lat}, {lon}) outside DEM bounds for window {metres} m; px={px}, py={py}, w={w}, h={h}")
                    ds = None
                    if dem_file.exists():
                        dem_file.unlink()
                    continue

                # Extract values at clamped pixel (safe)
                features = {k: float(v[clamped_py, clamped_px]) for k, v in attrs.items()}

                # Clean up
                ds = None
                if dem_file.exists():
                    dem_file.unlink()

                return features

            except Exception as e:
                last_error = e
                if dem_file.exists():
                    dem_file.unlink()
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
        
        if not self._download_elevation_data(region, str(dem_file)):
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
                dem_file.unlink()

    def _plot_dem(self, lat: float, lon: float, dem_file: str, 
                 save_plots: bool, output_dir: str) -> None:
        """Plot DEM elevation data."""
        ds = gdal.Open(dem_file)
        dem = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        h, w = dem.shape
        
        # Crop to window around point
        ys, xs = self._crop_window(gt, w, h, lat, lon)
        cropped = dem[ys, xs]
        
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
        # Load DEM and compute attributes
        attrs = self._load_dem_and_attributes(dem_file)
        slope = attrs['slope_degrees']
        
        ds = gdal.Open(dem_file)
        dem = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        h, w = dem.shape
        
        # Crop to window around point
        ys, xs = self._crop_window(gt, w, h, lat, lon)
        cropped = slope[ys, xs]
        
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
        if not self._download_elevation_data(region, str(dem_file)):
            raise RuntimeError(f"Failed to download DEM for 3D plot at ({lat}, {lon})")

        try:
            # Load raw DEM and attributes
            # Try GDAL read; fallback to PIL if needed
            dem = None
            ds = None
            try:
                ds = gdal.Open(str(dem_file))
                dem = ds.ReadAsArray()
                gt = ds.GetGeoTransform()
            except Exception:
                img = Image.open(str(dem_file))
                dem = np.array(img)
                # Construct a best-effort GeoTransform centered at point with pixel size ~10m
                # This is only used to compute a crop window around the center
                px_size = 10.0
                gt = (lon - (dem.shape[1] * px_size)/2.0, px_size, 0, lat + (dem.shape[0] * px_size)/2.0, 0, -px_size)
            h, w = dem.shape

            # Compute slope on full DEM, then crop consistent windows
            attrs = self._load_dem_and_attributes(str(dem_file))
            slope = attrs['slope_degrees']

            # Crop to window around the clicked point
            ys, xs = self._crop_window(gt, w, h, lat, lon, half_side_m=half_side_m)
            dem_c = dem[ys, xs]
            slope_c = slope[ys, xs]

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
            px_c, py_c = self._latlon_to_pixel(gt, lat, lon)
            # Clamp to cropped region center (0,0)
            if 0 <= px_c < w and 0 <= py_c < h:
                center_z = float(dem[int(py_c), int(px_c)])
            else:
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
                dem_file.unlink()

    def build_interactive_3d(self, lat: float, lon: float,
                              output_dir: str = "temp_dem",
                              half_side_m: int = 200) -> Tuple[go.Figure, str]:
        """
        Build an interactive 3D topographic Plotly figure (pan/zoom/rotate).
        Returns (figure, resolution_label).
        """
        Path(output_dir).mkdir(exist_ok=True)

        # Prefer ~10 m if possible by requesting a compact window
        region = self._offset(lat, lon, metres=max(2_000, half_side_m * 4))
        dem_file = Path(output_dir) / f"dem_3d_plotly_{lat:.5f}_{lon:.5f}.tif"
        if not self._download_elevation_data(region, str(dem_file)):
            raise RuntimeError(f"Failed to download DEM for interactive 3D at ({lat}, {lon})")

        try:
            # Read DEM (GDAL preferred)
            dem = None
            ds = None
            try:
                if GDAL_AVAILABLE:
                    ds = gdal.Open(str(dem_file))
                    dem = ds.ReadAsArray()
                    gt = ds.GetGeoTransform()
                else:
                    raise RuntimeError("GDAL not available")
            except Exception:
                img = Image.open(str(dem_file))
                dem = np.array(img)
                # Approximate degrees-per-pixel from ~10 m resolution at latitude
                px_m = 10.0
                deg_per_m_lat = 1.0 / 110540.0
                deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)) + 1e-9)
                xres = px_m * deg_per_m_lon
                yres = -px_m * deg_per_m_lat
                minx = lon - (dem.shape[1] * xres) / 2.0
                maxy = lat - (dem.shape[0] * yres) / 2.0  # yres is negative
                gt = (minx, xres, 0.0, maxy, 0.0, yres)

            h, w = dem.shape

            # Terrain attributes (slope for coloring)
            attrs = self._load_dem_and_attributes(str(dem_file))
            slope = attrs['slope_degrees']

            # Crop window
            ys, xs = self._crop_window(gt, w, h, lat, lon, half_side_m=half_side_m)
            dem_c = dem[ys, xs]
            slope_c = slope[ys, xs]

            # Downsample for performance
            size_y, size_x = dem_c.shape
            target = 200
            stride = int(max(1, np.ceil(max(size_x, size_y) / target)))
            if stride > 1:
                dem_c = dem_c[::stride, ::stride]
                slope_c = slope_c[::stride, ::stride]
                size_y, size_x = dem_c.shape

            # Build latitude/longitude grids corresponding to cropped/downsampled pixels
            xmin, xmax = xs.start, xs.stop
            ymin, ymax = ys.start, ys.stop
            x_idx = np.arange(xmin, xmax, stride)
            y_idx = np.arange(ymin, ymax, stride)
            lon_vals = gt[0] + x_idx * gt[1]
            lat_vals = gt[3] + y_idx * gt[5]
            X, Y = np.meshgrid(lon_vals, lat_vals)

            # Resolution label detection (approximate)
            try:
                m_per_deg_lat = 110540.0
                m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
                px_lat_m = abs(gt[5]) * m_per_deg_lat if ds is not None else 10.0
                px_lon_m = abs(gt[1]) * m_per_deg_lon if ds is not None else 10.0
                px_m = (px_lat_m + px_lon_m) / 2.0
                res_label = "10 m (1/3 arc-second)" if px_m <= 15.0 else "30 m (1 arc-second)"
            except Exception:
                res_label = "Unknown resolution"

            # Center point elevation
            px_c, py_c = self._latlon_to_pixel(gt, lat, lon)
            if 0 <= px_c < w and 0 <= py_c < h:
                center_z = float(dem[int(py_c), int(px_c)])
            else:
                center_z = float(np.nanmean(dem_c))

            # Build Plotly figure
            surface = go.Surface(x=X, y=Y, z=dem_c, surfacecolor=slope_c, colorscale='Plasma', colorbar=dict(title='Slope (°)'))
            marker = go.Scatter3d(x=[lon], y=[lat], z=[center_z], mode='markers', marker=dict(size=6, color='red'), name='Target')
            fig = go.Figure(data=[surface, marker])
            fig.update_scenes(xaxis_title='Longitude (°)', yaxis_title='Latitude (°)', zaxis_title='Elevation (m)')
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), title=f"Interactive 3D Topography – lat {lat:.5f}, lon {lon:.5f}")
            return fig, res_label
        finally:
            if 'ds' in locals() and ds is not None:
                ds = None
            if dem_file.exists():
                dem_file.unlink()

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