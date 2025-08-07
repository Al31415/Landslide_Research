"""
Data Collection Orchestrator
Coordinates all data collection modules for comprehensive data gathering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import time

# Import data collection modules
try:
    from .slope_data_collector import SlopeDataCollector
    from .ssurgo_data_collector import SSURGODataCollector
    from .weather_forecast_collector import WeatherForecastCollector
    from .meteostat_data_collector import MeteostatDataCollector
    ALL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some data collection modules not available: {e}")
    ALL_MODULES_AVAILABLE = False


class DataCollectionOrchestrator:
    """
    Orchestrates all data collection modules for comprehensive data gathering.
    
    This class coordinates the collection of:
    - Terrain/slope data
    - Soil data (SSURGO)
    - Weather forecast data
    - Historical weather data (Meteostat)
    """

    def __init__(self, output_dir: str = "collected_data"):
        """
        Initialize the data collection orchestrator.
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize collectors
        self.collectors = {}
        if ALL_MODULES_AVAILABLE:
            self.collectors['slope'] = SlopeDataCollector()
            self.collectors['soil'] = SSURGODataCollector()
            self.collectors['meteostat'] = MeteostatDataCollector()
            try:
                from .cmip_data_collector import CMIPDataCollector
                self.collectors['cmip'] = CMIPDataCollector()
            except ImportError:
                print("Warning: CMIP data collector not available")

    def collect_all_data(self, df: pd.DataFrame,
                        lat_col: str = 'Latitude',
                        lon_col: str = 'Longitude',
                        date_col: str = 'event_date',
                        modules: List[str] = None) -> pd.DataFrame:
        """
        Collect all available data types for the dataset.
        
        Args:
            df: Input DataFrame
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            date_col: Name of date column
            modules: List of modules to use (default: all available)
            
        Returns:
            DataFrame with all collected data
        """
        if modules is None:
            modules = list(self.collectors.keys())
        
        print("="*80)
        print("COMPREHENSIVE DATA COLLECTION")
        print("="*80)
        
        result_df = df.copy()
        collection_summary = {}
        
        start_time = time.time()
        
        # Collect terrain/slope data
        if 'slope' in modules and 'slope' in self.collectors:
            print("\n1. Collecting terrain/slope data...")
            try:
                result_df = self.collectors['slope'].collect_slope_data_for_dataset(
                    result_df, lat_col, lon_col, self.output_dir / "slope_data"
                )
                collection_summary['slope'] = 'success'
                print("Terrain/slope data collected successfully")
            except Exception as e:
                print(f"Error collecting terrain data: {e}")
                collection_summary['slope'] = 'failed'
        
        # Collect soil data
        if 'soil' in modules and 'soil' in self.collectors:
            print("\n2. Collecting soil data (SSURGO)...")
            try:
                result_df = self.collectors['soil'].collect_soil_data_for_dataset(
                    result_df, lat_col, lon_col, aggregate=True
                )
                collection_summary['soil'] = 'success'
                print("Soil data collected successfully")
            except Exception as e:
                print(f"Error collecting soil data: {e}")
                collection_summary['soil'] = 'failed'
        
        # Collect historical weather data
        if 'meteostat' in modules and 'meteostat' in self.collectors:
            print("\n3. Collecting historical weather data (Meteostat)...")
            try:
                result_df = self.collectors['meteostat'].fill_missing_precipitation_data(
                    result_df, lat_col, lon_col, date_col
                )
                collection_summary['meteostat'] = 'success'
                print("Historical weather data collected successfully")
            except Exception as e:
                print(f"Error collecting historical weather data: {e}")
                collection_summary['meteostat'] = 'failed'
        
        # Collect CMIP6 climate data
        if 'cmip' in modules and 'cmip' in self.collectors:
            print("\n4. Collecting CMIP6 climate data...")
            try:
                result_df = self.collectors['cmip'].create_forecast_dataset(
                    result_df, lat_col, lon_col, date_col
                )
                collection_summary['cmip'] = 'success'
                print("CMIP6 climate data collected successfully")
            except Exception as e:
                print(f"Error collecting CMIP6 data: {e}")
                collection_summary['cmip'] = 'failed'
        
        # Save results
        self._save_collected_data(result_df, collection_summary)
        
        # Print summary
        end_time = time.time()
        self._print_collection_summary(collection_summary, end_time - start_time)
        
        return result_df

    def collect_weather_forecast_data(self, df: pd.DataFrame,
                                    climate_file: str,
                                    parameter_name: str = 'pr',
                                    lat_col: str = 'Latitude',
                                    lon_col: str = 'Longitude',
                                    date_col: str = 'event_date') -> pd.DataFrame:
        """
        Collect weather forecast data using climate dataset.
        
        Args:
            df: Input DataFrame
            climate_file: Path to climate dataset file
            parameter_name: Climate parameter name
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            date_col: Name of date column
            
        Returns:
            DataFrame with weather forecast data
        """
        print("\nCollecting weather forecast data...")
        
        try:
            from .weather_forecast_collector import collect_weather_forecast_data
            
            forecast_df = collect_weather_forecast_data(
                df, climate_file, parameter_name
            )
            
            # Save forecast data
            forecast_file = self.output_dir / "weather_forecast_data.csv"
            forecast_df.to_csv(forecast_file, index=False)
            print(f"✓ Weather forecast data saved to {forecast_file}")
            
            return forecast_df
            
        except Exception as e:
            print(f"✗ Error collecting weather forecast data: {e}")
            return df.copy()

    def collect_specific_data_type(self, df: pd.DataFrame,
                                 data_type: str,
                                 **kwargs) -> pd.DataFrame:
        """
        Collect specific type of data.
        
        Args:
            df: Input DataFrame
            data_type: Type of data to collect ('slope', 'soil', 'meteostat', 'weather_forecast')
            **kwargs: Additional arguments for specific collectors
            
        Returns:
            DataFrame with collected data
        """
        if data_type not in self.collectors:
            raise ValueError(f"Unknown data type: {data_type}")
        
        print(f"Collecting {data_type} data...")
        
        collector = self.collectors[data_type]
        
        if data_type == 'slope':
            return collector.collect_slope_data_for_dataset(df, **kwargs)
        elif data_type == 'soil':
            return collector.collect_soil_data_for_dataset(df, **kwargs)
        elif data_type == 'meteostat':
            return collector.fill_missing_precipitation_data(df, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def validate_collected_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of collected data.
        
        Args:
            df: DataFrame with collected data
            
        Returns:
            Dictionary with validation results
        """
        print("\nValidating collected data...")
        
        validation_results = {}
        
        # Validate terrain data
        terrain_cols = [col for col in df.columns if col.startswith('terrain_')]
        if terrain_cols:
            validation_results['terrain'] = self._validate_numeric_data(df[terrain_cols])
        
        # Validate soil data
        soil_cols = [col for col in df.columns if col.startswith('soil_')]
        if soil_cols:
            validation_results['soil'] = self._validate_numeric_data(df[soil_cols])
        
        # Validate precipitation data
        precip_cols = [col for col in df.columns if 'prcp' in col.lower()]
        if precip_cols:
            validation_results['precipitation'] = self._validate_numeric_data(df[precip_cols])
        
        return validation_results

    def _validate_numeric_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate numeric data columns."""
        validation = {}
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                col_data = data[col].dropna()
                validation[col] = {
                    'count': len(col_data),
                    'missing': data[col].isna().sum(),
                    'min': float(col_data.min()) if len(col_data) > 0 else None,
                    'max': float(col_data.max()) if len(col_data) > 0 else None,
                    'mean': float(col_data.mean()) if len(col_data) > 0 else None,
                    'std': float(col_data.std()) if len(col_data) > 0 else None
                }
        
        return validation

    def _save_collected_data(self, df: pd.DataFrame, collection_summary: Dict[str, str]):
        """Save collected data to files."""
        # Save main dataset
        main_file = self.output_dir / "collected_data_complete.csv"
        df.to_csv(main_file, index=False)
        print(f"\nComplete dataset saved to {main_file}")
        
        # Save collection summary
        summary_file = self.output_dir / "collection_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Data Collection Summary\n")
            f.write("="*50 + "\n")
            for module, status in collection_summary.items():
                f.write(f"{module}: {status}\n")
        print(f"Collection summary saved to {summary_file}")

    def _print_collection_summary(self, collection_summary: Dict[str, str], duration: float):
        """Print collection summary."""
        print("\n" + "="*80)
        print("COLLECTION SUMMARY")
        print("="*80)
        
        for module, status in collection_summary.items():
            status_text = "SUCCESS" if status == 'success' else "FAILED" if status == 'failed' else "SKIPPED"
            print(f"{module}: {status_text}")
        
        print(f"\nTotal collection time: {duration:.2f} seconds")
        print(f"Output directory: {self.output_dir}")


def run_complete_data_collection(input_file: str,
                               output_dir: str = "collected_data",
                               modules: List[str] = None) -> pd.DataFrame:
    """
    Run complete data collection pipeline.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory for output files
        modules: List of modules to use (default: all available)
        
    Returns:
        DataFrame with all collected data
    """
    # Load input data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator(output_dir)
    
    # Collect all data
    result_df = orchestrator.collect_all_data(df, modules=modules)
    
    # Validate collected data
    validation_results = orchestrator.validate_collected_data(result_df)
    
    print("\nData collection completed successfully!")
    return result_df


if __name__ == "__main__":
    # Example usage
    print("Data Collection Orchestrator Example")
    print("="*50)
    
    # This would require an actual input file
    # Example of how to use the orchestrator:
    
    # orchestrator = DataCollectionOrchestrator("example_output")
    # 
    # # Load your data
    # df = pd.read_csv("your_input_data.csv")
    # 
    # # Collect all data
    # result_df = orchestrator.collect_all_data(df)
    # 
    # # Or collect specific data types
    # slope_df = orchestrator.collect_specific_data_type(df, 'slope')
    # soil_df = orchestrator.collect_specific_data_type(df, 'soil')
    
    print("Data collection orchestrator initialized successfully!")
    print("Use run_complete_data_collection() to run the full pipeline.") 