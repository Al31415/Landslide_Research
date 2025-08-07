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
import richdem as rd
import shapefile
from osgeo import gdal
from shapely.geometry import Point, Polygon
from PIL import Image


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

    def _compute_terrain_attributes(self, dem: rd.rdarray) -> Dict[str, np.ndarray]:
        """
        Compute terrain attributes from DEM.
        
        Args:
            dem: RichDEM array of elevation data
            
        Returns:
            Dictionary containing terrain attributes
        """
        return {
            "slope_degrees": rd.TerrainAttribute(dem, "slope_degrees"),
            "aspect": rd.TerrainAttribute(dem, "aspect"),
            "planform_curvature": rd.TerrainAttribute(dem, "planform_curvature"),
            "profile_curvature": rd.TerrainAttribute(dem, "profile_curvature"),
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
            imarray_rd = rd.rdarray(imarray, no_data=-9999)
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
        
        # Download DEM data
        region = self._offset(lat, lon)
        dem_file = Path(output_dir) / f"dem_{lat:.5f}_{lon:.5f}.tif"
        
        if not self._download_elevation_data(region, str(dem_file)):
            raise RuntimeError(f"Failed to download DEM for point ({lat}, {lon})")
        
        try:
            # Load DEM and compute attributes
            attrs = self._load_dem_and_attributes(str(dem_file))
            
            # Get pixel coordinates
            ds = gdal.Open(str(dem_file))
            gt = ds.GetGeoTransform()
            h, w = ds.ReadAsArray().shape
            px, py = self._latlon_to_pixel(gt, lat, lon)
            
            # Check bounds
            if not (0 <= px < w and 0 <= py < h):
                raise ValueError(f"Point ({lat}, {lon}) outside DEM bounds")
            
            # Extract values at point
            features = {k: float(v[py, px]) for k, v in attrs.items()}
            
            # Clean up
            ds = None
            if dem_file.exists():
                dem_file.unlink()
            
            return features
            
        except Exception as e:
            # Clean up on error
            if dem_file.exists():
                dem_file.unlink()
            raise e

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