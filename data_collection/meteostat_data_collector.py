"""
Meteostat Data Collector Module
Collects historical weather data using the Meteostat API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
from tqdm import tqdm

try:
    from meteostat import Point, Daily
    METEOSTAT_AVAILABLE = True
except ImportError:
    METEOSTAT_AVAILABLE = False
    warnings.warn("Meteostat not available. Install with: pip install meteostat")


class MeteostatDataCollector:
    """
    Collect historical weather data using Meteostat API.
    
    This module provides methods to collect precipitation and other weather
    data for given coordinates and date ranges.
    """

    def __init__(self, days_back: int = 400, rate_limit_delay: float = 0.1):
        """
        Initialize the Meteostat data collector.
        
        Args:
            days_back: Number of days to look back for data collection
            rate_limit_delay: Delay between API calls in seconds
        """
        if not METEOSTAT_AVAILABLE:
            raise ImportError("Meteostat is not available. Install with: pip install meteostat")
        
        self.days_back = days_back
        self.rate_limit_delay = rate_limit_delay

    def get_precipitation_data(self, lat: float, lon: float, event_date: datetime) -> Dict[str, float]:
        """
        Get precipitation data for a specific location and date.
        
        Args:
            lat: Latitude
            lon: Longitude
            event_date: Date for which to collect data
            
        Returns:
            Dictionary with precipitation statistics
        """
        try:
            # Create point location
            location = Point(lat, lon)
            
            # Calculate date range
            end_date = event_date
            start_date = end_date - timedelta(days=self.days_back)
            
            # Get daily data
            data = Daily(location, start_date, end_date)
            data = data.fetch()
            
            if data.empty:
                warnings.warn(f"No precipitation data found for ({lat}, {lon}) on {event_date}")
                return self._get_empty_precipitation_stats()
            
            # Calculate precipitation statistics
            stats = self._calculate_precipitation_statistics(data)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return stats
            
        except Exception as e:
            warnings.warn(f"Error collecting precipitation data for ({lat}, {lon}): {e}")
            return self._get_empty_precipitation_stats()

    def _calculate_precipitation_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate precipitation statistics from daily data.
        
        Args:
            data: DataFrame with daily precipitation data
            
        Returns:
            Dictionary with precipitation statistics
        """
        if data.empty or 'prcp' not in data.columns:
            return self._get_empty_precipitation_stats()
        
        # Get precipitation column
        prcp = data['prcp'].fillna(0)
        
        # Calculate statistics
        stats = {
            'max_1_day_prcp': float(prcp.max()) if not prcp.empty else 0.0,
            'max_3_day_prcp': float(prcp.rolling(window=3, min_periods=1).sum().max()) if len(prcp) >= 3 else 0.0,
            'max_7_day_prcp': float(prcp.rolling(window=7, min_periods=1).sum().max()) if len(prcp) >= 7 else 0.0,
            'max_14_day_prcp': float(prcp.rolling(window=14, min_periods=1).sum().max()) if len(prcp) >= 14 else 0.0,
            'avg_30_day_prcp': float(prcp.tail(30).mean()) if len(prcp) >= 30 else 0.0,
            'avg_60_day_prcp': float(prcp.tail(60).mean()) if len(prcp) >= 60 else 0.0,
            'avg_90_day_prcp': float(prcp.tail(90).mean()) if len(prcp) >= 90 else 0.0,
            'avg_365_day_prcp': float(prcp.tail(365).mean()) if len(prcp) >= 365 else 0.0
        }
        
        return stats

    def _get_empty_precipitation_stats(self) -> Dict[str, float]:
        """Return empty precipitation statistics."""
        return {
            'max_1_day_prcp': 0.0,
            'max_3_day_prcp': 0.0,
            'max_7_day_prcp': 0.0,
            'max_14_day_prcp': 0.0,
            'avg_30_day_prcp': 0.0,
            'avg_60_day_prcp': 0.0,
            'avg_90_day_prcp': 0.0,
            'avg_365_day_prcp': 0.0
        }

    def collect_precipitation_data_batch(self, coordinates: List[Tuple[float, float, datetime]]) -> pd.DataFrame:
        """
        Collect precipitation data for multiple coordinates and dates.
        
        Args:
            coordinates: List of (lat, lon, date) tuples
            
        Returns:
            DataFrame with precipitation data for all coordinates
        """
        results = []
        
        for i, (lat, lon, event_date) in enumerate(tqdm(coordinates, desc="Collecting precipitation data")):
            try:
                stats = self.get_precipitation_data(lat, lon, event_date)
                stats['latitude'] = lat
                stats['longitude'] = lon
                stats['event_date'] = event_date
                results.append(stats)
                
            except Exception as e:
                warnings.warn(f"Failed to collect data for coordinate {i+1}: ({lat}, {lon}): {e}")
                # Add empty row
                empty_stats = self._get_empty_precipitation_stats()
                empty_stats.update({
                    'latitude': lat,
                    'longitude': lon,
                    'event_date': event_date
                })
                results.append(empty_stats)
        
        return pd.DataFrame(results)

    def fill_missing_precipitation_data(self, df: pd.DataFrame,
                                      lat_col: str = 'Latitude',
                                      lon_col: str = 'Longitude',
                                      date_col: str = 'event_date',
                                      precip_columns: List[str] = None) -> pd.DataFrame:
        """
        Fill missing precipitation data in a dataset.
        
        Args:
            df: Input DataFrame
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            date_col: Name of date column
            precip_columns: List of precipitation columns to fill
            
        Returns:
            DataFrame with filled precipitation data
        """
        if precip_columns is None:
            precip_columns = [
                'max_1_day_prcp', 'max_3_day_prcp', 'max_7_day_prcp', 'max_14_day_prcp',
                'avg_30_day_prcp', 'avg_60_day_prcp', 'avg_90_day_prcp', 'avg_365_day_prcp'
            ]
        
        # Create copy of original data
        result_df = df.copy()
        
        # Find rows with missing precipitation data
        missing_mask = result_df[precip_columns].isna().any(axis=1)
        missing_rows = result_df[missing_mask]
        
        if missing_rows.empty:
            print("No missing precipitation data found.")
            return result_df
        
        print(f"Found {len(missing_rows)} rows with missing precipitation data.")
        
        # Prepare coordinates for data collection
        coordinates = []
        for _, row in missing_rows.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            event_date = row[date_col]
            
            if pd.notna(lat) and pd.notna(lon) and pd.notna(event_date):
                # Convert date to datetime if needed
                if isinstance(event_date, str):
                    event_date = pd.to_datetime(event_date)
                coordinates.append((lat, lon, event_date))
        
        if not coordinates:
            print("No valid coordinates found for data collection.")
            return result_df
        
        # Collect precipitation data
        precip_data = self.collect_precipitation_data_batch(coordinates)
        
        # Update the original DataFrame
        for _, precip_row in precip_data.iterrows():
            # Find matching row in original data
            mask = ((result_df[lat_col] == precip_row['latitude']) &
                   (result_df[lon_col] == precip_row['longitude']) &
                   (result_df[date_col] == precip_row['event_date']))
            
            if mask.any():
                for col in precip_columns:
                    if col in precip_row:
                        result_df.loc[mask, col] = precip_row[col]
        
        # Report results
        still_missing = result_df[precip_columns].isna().sum().sum()
        print(f"Precipitation data collection completed. {still_missing} values still missing.")
        
        return result_df

    def create_complete_meteostat_dataset(self, df: pd.DataFrame,
                                        lat_col: str = 'Latitude',
                                        lon_col: str = 'Longitude',
                                        date_col: str = 'event_date',
                                        precip_columns: List[str] = None) -> pd.DataFrame:
        """
        Create a complete dataset with all precipitation data from Meteostat.
        
        Args:
            df: Input DataFrame
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            date_col: Name of date column
            precip_columns: List of precipitation columns to replace
            
        Returns:
            DataFrame with all precipitation data from Meteostat
        """
        if precip_columns is None:
            precip_columns = [
                'max_1_day_prcp', 'max_3_day_prcp', 'max_7_day_prcp', 'max_14_day_prcp',
                'avg_30_day_prcp', 'avg_60_day_prcp', 'avg_90_day_prcp', 'avg_365_day_prcp'
            ]
        
        print(f"Creating complete Meteostat dataset for {len(df)} rows...")
        
        # Prepare coordinates for all rows
        coordinates = []
        for _, row in df.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            event_date = row[date_col]
            
            if pd.notna(lat) and pd.notna(lon) and pd.notna(event_date):
                # Convert date to datetime if needed
                if isinstance(event_date, str):
                    event_date = pd.to_datetime(event_date)
                coordinates.append((lat, lon, event_date))
        
        if not coordinates:
            print("No valid coordinates found.")
            return df.copy()
        
        # Collect precipitation data for all coordinates
        precip_data = self.collect_precipitation_data_batch(coordinates)
        
        # Create result DataFrame
        result_df = df.copy()
        
        # Update precipitation columns
        for _, precip_row in precip_data.iterrows():
            # Find matching row in original data
            mask = ((result_df[lat_col] == precip_row['latitude']) &
                   (result_df[lon_col] == precip_row['longitude']) &
                   (result_df[date_col] == precip_row['event_date']))
            
            if mask.any():
                for col in precip_columns:
                    if col in precip_row:
                        result_df.loc[mask, col] = precip_row[col]
        
        # Report results
        missing_counts = result_df[precip_columns].isna().sum()
        print("\nPrecipitation data collection summary:")
        for col in precip_columns:
            total = len(result_df)
            missing = missing_counts[col]
            success_rate = ((total - missing) / total) * 100
            print(f"  {col}: {total - missing}/{total} ({success_rate:.1f}%)")
        
        return result_df

    def validate_data_quality(self, df: pd.DataFrame, precip_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate the quality of collected precipitation data.
        
        Args:
            df: DataFrame with precipitation data
            precip_columns: List of precipitation columns to validate
            
        Returns:
            Dictionary with validation results
        """
        if precip_columns is None:
            precip_columns = [
                'max_1_day_prcp', 'max_3_day_prcp', 'max_7_day_prcp', 'max_14_day_prcp',
                'avg_30_day_prcp', 'avg_60_day_prcp', 'avg_90_day_prcp', 'avg_365_day_prcp'
            ]
        
        validation_results = {}
        
        for col in precip_columns:
            if col not in df.columns:
                validation_results[col] = {'status': 'missing_column'}
                continue
            
            data = df[col].dropna()
            
            if len(data) == 0:
                validation_results[col] = {'status': 'no_data'}
                continue
            
            validation_results[col] = {
                'status': 'valid',
                'count': len(data),
                'missing': df[col].isna().sum(),
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'zero_count': int((data == 0).sum()),
                'negative_count': int((data < 0).sum())
            }
        
        return validation_results


def collect_meteostat_data_for_dataset(df: pd.DataFrame,
                                     lat_col: str = 'Latitude',
                                     lon_col: str = 'Longitude',
                                     date_col: str = 'event_date',
                                     fill_missing_only: bool = True) -> pd.DataFrame:
    """
    Convenience function to collect Meteostat data for a dataset.
    
    Args:
        df: Input DataFrame
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        date_col: Name of date column
        fill_missing_only: If True, only fill missing values; if False, replace all
        
    Returns:
        DataFrame with Meteostat precipitation data
    """
    collector = MeteostatDataCollector()
    
    if fill_missing_only:
        return collector.fill_missing_precipitation_data(df, lat_col, lon_col, date_col)
    else:
        return collector.create_complete_meteostat_dataset(df, lat_col, lon_col, date_col)


if __name__ == "__main__":
    # Example usage
    if not METEOSTAT_AVAILABLE:
        print("Meteostat not available. Install with: pip install meteostat")
    else:
        collector = MeteostatDataCollector()
        
        # Test with single point
        lat, lon = 37.54189, -120.96683
        event_date = datetime(2020, 1, 15)
        
        try:
            precip_data = collector.get_precipitation_data(lat, lon, event_date)
            print(f"Precipitation data for ({lat}, {lon}) on {event_date}:")
            for key, value in precip_data.items():
                print(f"  {key}: {value:.2f}")
        except Exception as e:
            print(f"Error collecting precipitation data: {e}")
        
        # Test batch collection
        coordinates = [
            (37.54189, -120.96683, datetime(2020, 1, 15)),
            (37.9014, -121.1895, datetime(2020, 1, 15))
        ]
        
        try:
            batch_data = collector.collect_precipitation_data_batch(coordinates)
            print(f"\nBatch collection results:")
            print(batch_data)
        except Exception as e:
            print(f"Error in batch collection: {e}") 