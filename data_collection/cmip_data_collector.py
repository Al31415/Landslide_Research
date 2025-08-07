"""
CMIP Data Collector Module
Collects and processes CMIP6 climate model data for stability analysis.
"""

import xarray as xr
import pandas as pd
import numpy as np
import cftime
import re
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import warnings


class CMIPDataCollector:
    """
    Collect and process CMIP6 climate model data.
    
    This module handles CMIP6 NetCDF files and processes them for
    stability analysis, including historical and future scenarios.
    """

    def __init__(self, data_dir: str = "data", model: str = "CESM2"):
        """
        Initialize the CMIP data collector.
        
        Args:
            data_dir: Directory containing CMIP6 NetCDF files
            model: Climate model name (default: CESM2)
        """
        self.data_dir = Path(data_dir)
        self.model = model
        self.scenarios = ['hist', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
        
        # File patterns for different scenarios
        self.file_patterns = {
            "hist": f"pr_Amon_{model}_historical_r1i1p1f1_gn_*.nc",
            "ssp126": f"pr_Amon_{model}_ssp126_r4i1p1f1_gn_*.nc",
            "ssp245": f"pr_Amon_{model}_ssp245_r4i1p1f1_gn_*.nc",
            "ssp370": f"pr_Amon_{model}_ssp370_r4i1p1f1_gn_*.nc",
            "ssp585": f"pr_Amon_{model}_ssp585_r4i1p1f1_gn_*.nc"
        }
        
        # Scenario metadata
        self.scenario_metadata = {
            "ssp126": ["ssp126", "r4"],
            "ssp245": ["ssp245", "r4"],
            "ssp370": ["ssp370", "r4"], 
            "ssp585": ["ssp585", "r4"]
        }

    def find_cmip_files(self) -> Dict[str, str]:
        """
        Find CMIP6 NetCDF files in the data directory.
        
        Returns:
            Dictionary mapping scenario names to file paths
        """
        files = {}
        
        for scenario, pattern in self.file_patterns.items():
            file_path = list(self.data_dir.glob(pattern))
            if file_path:
                files[scenario] = str(file_path[0])
                print(f"Found {scenario} file: {file_path[0].name}")
            else:
                print(f"Warning: No file found for scenario {scenario}")
        
        return files

    def load_cmip_data(self, scenario: str) -> Optional[xr.Dataset]:
        """
        Load CMIP6 data for a specific scenario.
        
        Args:
            scenario: Scenario name (hist, ssp126, ssp245, ssp370, ssp585)
            
        Returns:
            xarray Dataset or None if file not found
        """
        files = self.find_cmip_files()
        
        if scenario not in files:
            print(f"No file found for scenario {scenario}")
            return None
        
        try:
            print(f"Loading {scenario} data from {files[scenario]}")
            dataset = xr.open_dataset(files[scenario])
            return dataset
        except Exception as e:
            print(f"Error loading {scenario} data: {e}")
            return None

    def build_monthly_series(self, scenario: str) -> Optional[xr.DataArray]:
        """
        Build monthly precipitation series for a scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            Monthly precipitation DataArray
        """
        dataset = self.load_cmip_data(scenario)
        if dataset is None:
            return None
        
        try:
            # Extract precipitation data
            if 'pr' in dataset:
                pr_data = dataset['pr']
                
                # Convert units if needed (kg m-2 s-1 to mm/month)
                if 'units' in pr_data.attrs:
                    if 'kg m-2 s-1' in pr_data.attrs['units']:
                        # Convert to mm/month (assuming monthly data)
                        seconds_per_month = 30.44 * 24 * 3600  # average seconds per month
                        pr_data = pr_data * seconds_per_month
                        pr_data.attrs['units'] = 'mm/month'
                
                return pr_data
            else:
                print(f"No precipitation data found in {scenario} dataset")
                return None
                
        except Exception as e:
            print(f"Error building monthly series for {scenario}: {e}")
            return None

    def stitch_monthly(self, hist_data: xr.DataArray, future_data: xr.DataArray) -> xr.DataArray:
        """
        Stitch historical and future monthly data together.
        
        Args:
            hist_data: Historical monthly data
            future_data: Future monthly data
            
        Returns:
            Combined monthly data
        """
        try:
            # Find the overlap point (usually 2015)
            hist_end = hist_data.time.max()
            future_start = future_data.time.min()
            
            # Remove overlap from historical data
            hist_trimmed = hist_data.sel(time=slice(None, hist_end))
            
            # Combine datasets
            combined = xr.concat([hist_trimmed, future_data], dim='time')
            
            return combined
            
        except Exception as e:
            print(f"Error stitching monthly data: {e}")
            return future_data

    def rolling_sum(self, data: xr.DataArray, months: int = 12) -> xr.DataArray:
        """
        Calculate rolling sum over specified number of months.
        
        Args:
            data: Monthly precipitation data
            months: Number of months for rolling window
            
        Returns:
            Rolling sum DataArray
        """
        try:
            # Calculate rolling sum
            rolling = data.rolling(time=months, center=True).sum()
            
            # Add metadata
            rolling.attrs['description'] = f'{months}-month rolling precipitation sum'
            rolling.attrs['units'] = 'mm'
            
            return rolling
            
        except Exception as e:
            print(f"Error calculating rolling sum: {e}")
            return data

    def process_all_scenarios(self, roll_months: int = 12) -> Dict[str, xr.DataArray]:
        """
        Process all CMIP6 scenarios and create rolling precipitation data.
        
        Args:
            roll_months: Number of months for rolling window
            
        Returns:
            Dictionary of rolling precipitation data for each scenario
        """
        print("Processing all CMIP6 scenarios...")
        
        rolling_data = {}
        
        # Process historical data first
        hist_data = self.build_monthly_series('hist')
        if hist_data is None:
            print("Warning: Could not load historical data")
            return rolling_data
        
        # Process each future scenario
        for scenario in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
            print(f"Processing scenario {scenario}...")
            
            try:
                # Load future scenario data
                future_data = self.build_monthly_series(scenario)
                if future_data is None:
                    continue
                
                # Stitch with historical data
                combined_data = self.stitch_monthly(hist_data, future_data)
                
                # Calculate rolling sum
                rolling_sum = self.rolling_sum(combined_data, roll_months)
                
                # Load data into memory
                rolling_sum = rolling_sum.load()
                
                rolling_data[scenario] = rolling_sum
                
                print(f"Completed processing {scenario}")
                
            except Exception as e:
                print(f"Error processing {scenario}: {e}")
                continue
        
        return rolling_data

    def extract_data_at_location(self, rolling_data: Dict[str, xr.DataArray],
                               lat: float, lon: float, date: datetime,
                               scenario: str = "ssp245") -> Optional[float]:
        """
        Extract precipitation data at specific location and date.
        
        Args:
            rolling_data: Dictionary of rolling precipitation data
            lat: Latitude
            lon: Longitude
            date: Date for extraction
            scenario: Scenario to use
            
        Returns:
            Precipitation value or None if not available
        """
        if scenario not in rolling_data:
            print(f"Scenario {scenario} not found in rolling data")
            return None
        
        try:
            # Convert longitude to 0-360 range if needed
            lon360 = lon if lon >= 0 else lon + 360
            
            # Find nearest grid point
            data = rolling_data[scenario]
            
            # Interpolate to exact location
            point_data = data.sel(
                lat=lat, 
                lon=lon360, 
                method='nearest'
            )
            
            # Find nearest time
            time_data = point_data.sel(time=date, method='nearest')
            
            # Extract value
            value = float(time_data.values)
            
            if np.isnan(value):
                print(f"NaN value returned for {scenario} data at lat={lat}, lon={lon360}, date={date}")
                return None
            
            return value
            
        except Exception as e:
            print(f"Error extracting {scenario} data: {e}")
            return None

    def calculate_precipitation_metrics(self, rolling_data: Dict[str, xr.DataArray],
                                      lat: float, lon: float, event_date: datetime,
                                      scenario: str = "ssp245") -> Dict[str, float]:
        """
        Calculate precipitation metrics for a location and date.
        
        Args:
            rolling_data: Dictionary of rolling precipitation data
            lat: Latitude
            lon: Longitude
            event_date: Event date
            scenario: Scenario to use
            
        Returns:
            Dictionary of precipitation metrics
        """
        metrics = {}
        
        # Get data for the event date
        event_value = self.extract_data_at_location(
            rolling_data, lat, lon, event_date, scenario
        )
        
        if event_value is not None:
            metrics[f'{scenario}_event_precipitation'] = event_value
        
        # Calculate additional metrics if needed
        # This could include seasonal averages, trends, etc.
        
        return metrics

    def create_forecast_dataset(self, rolling_data: Dict[str, xr.DataArray],
                              df: pd.DataFrame,
                              lat_col: str = 'Latitude',
                              lon_col: str = 'Longitude',
                              date_col: str = 'event_date') -> pd.DataFrame:
        """
        Create forecast dataset with CMIP6 data for all locations.
        
        Args:
            rolling_data: Dictionary of rolling precipitation data
            df: Input DataFrame with coordinates and dates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            date_col: Name of date column
            
        Returns:
            DataFrame with CMIP6 forecast data
        """
        print("Creating forecast dataset with CMIP6 data...")
        
        result_df = df.copy()
        
        # Process each scenario
        for scenario in rolling_data.keys():
            print(f"Processing {scenario} scenario...")
            
            scenario_metrics = []
            
            for idx, row in result_df.iterrows():
                lat = row[lat_col]
                lon = row[lon_col]
                event_date = row[date_col]
                
                if pd.isna(lat) or pd.isna(lon) or pd.isna(event_date):
                    scenario_metrics.append({})
                    continue
                
                # Convert date to datetime if needed
                if isinstance(event_date, str):
                    event_date = pd.to_datetime(event_date)
                
                # Calculate metrics for this location and date
                metrics = self.calculate_precipitation_metrics(
                    rolling_data, lat, lon, event_date, scenario
                )
                
                scenario_metrics.append(metrics)
            
            # Add scenario data to DataFrame
            for metric_name in scenario_metrics[0].keys():
                values = [m.get(metric_name, np.nan) for m in scenario_metrics]
                result_df[metric_name] = values
        
        return result_df

    def save_forecast_data(self, rolling_data: Dict[str, xr.DataArray],
                          output_dir: str = "output") -> None:
        """
        Save forecast data to CSV files.
        
        Args:
            rolling_data: Dictionary of rolling precipitation data
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for scenario, data in rolling_data.items():
            # Convert to DataFrame
            df = data.to_dataframe().reset_index()
            
            # Save to CSV
            output_file = output_path / f"forecast_{scenario}_2025-2100.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved {scenario} forecast data to {output_file}")


def collect_cmip_data_for_dataset(df: pd.DataFrame,
                                data_dir: str = "data",
                                lat_col: str = 'Latitude',
                                lon_col: str = 'Longitude',
                                date_col: str = 'event_date') -> pd.DataFrame:
    """
    Convenience function to collect CMIP6 data for a dataset.
    
    Args:
        df: Input DataFrame
        data_dir: Directory containing CMIP6 files
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        date_col: Name of date column
        
    Returns:
        DataFrame with CMIP6 forecast data
    """
    # Initialize collector
    collector = CMIPDataCollector(data_dir)
    
    # Process all scenarios
    rolling_data = collector.process_all_scenarios()
    
    if not rolling_data:
        print("No CMIP6 data processed successfully")
        return df.copy()
    
    # Create forecast dataset
    result_df = collector.create_forecast_dataset(
        rolling_data, df, lat_col, lon_col, date_col
    )
    
    # Save forecast data
    collector.save_forecast_data(rolling_data)
    
    return result_df


if __name__ == "__main__":
    # Example usage
    print("CMIP Data Collector Example")
    print("="*50)
    
    # Initialize collector
    collector = CMIPDataCollector("data")
    
    # Find available files
    files = collector.find_cmip_files()
    print(f"Found {len(files)} CMIP6 files")
    
    # Process scenarios
    rolling_data = collector.process_all_scenarios()
    print(f"Processed {len(rolling_data)} scenarios")
    
    # Example data extraction
    if 'ssp245' in rolling_data:
        value = collector.extract_data_at_location(
            rolling_data, 37.0, -120.0, datetime(2025, 1, 15), 'ssp245'
        )
        print(f"SSP245 precipitation at (37, -120) on 2025-01-15: {value}") 