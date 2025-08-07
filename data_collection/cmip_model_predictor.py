"""
CMIP Model Predictor Module
Uses CMIP6 climate forecasts to predict stability changes from 2025-2100.
"""

import pandas as pd
import numpy as np
import xarray as xr
import cftime
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
from pathlib import Path


class CMIPModelPredictor:
    """
    Predict stability changes using CMIP6 climate forecasts.
    
    This module trains models on historical data and applies them to
    future climate scenarios to predict stability changes.
    """

    def __init__(self, cmip_data: Dict[str, xr.DataArray] = None):
        """
        Initialize the CMIP model predictor.
        
        Args:
            cmip_data: Dictionary of CMIP6 rolling precipitation data
        """
        self.cmip_data = cmip_data or {}
        self.model = None
        self.original_predictions = None
        self.feature_columns = None
        self.scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    def train_stability_model(self, df: pd.DataFrame,
                            feature_columns: List[str],
                            target_column: str = 'Stability',
                            test_size: float = 0.2,
                            random_state: int = 42) -> Dict[str, Any]:
        """
        Train a Random Forest model for stability prediction.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary containing model and training results
        """
        print("Training stability prediction model...")
        
        try:
            # Prepare data
            X = df[feature_columns].dropna()
            y = df[target_column].loc[X.index]
            
            if len(X) == 0:
                raise ValueError("No valid data for training after removing NaN values")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store original predictions for the full dataset
            self.original_predictions = self.model.predict(df[feature_columns].fillna(0))
            self.feature_columns = feature_columns
            
            results = {
                'model': self.model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'feature_importance': dict(zip(feature_columns, self.model.feature_importances_)),
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            print(f"Model training completed:")
            print(f"  Training accuracy: {train_accuracy:.4f}")
            print(f"  Test accuracy: {test_accuracy:.4f}")
            print(f"  CV accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
            
            return results
            
        except Exception as e:
            print(f"Error training stability model: {e}")
            return {}

    def extract_cmip_data_for_location(self, lat: float, lon: float, 
                                     event_date: datetime,
                                     scenario: str = "ssp245") -> Dict[str, float]:
        """
        Extract CMIP6 data for a specific location and date.
        
        Args:
            lat: Latitude
            lon: Longitude
            event_date: Event date
            scenario: CMIP6 scenario
            
        Returns:
            Dictionary of CMIP6 data for the location
        """
        if scenario not in self.cmip_data:
            print(f"Scenario {scenario} not found in CMIP data")
            return {}
        
        try:
            # Convert longitude to 0-360 range if needed
            lon360 = lon if lon >= 0 else lon + 360
            
            # Get data for the scenario
            data = self.cmip_data[scenario]
            
            # Extract data at location and date
            point_data = data.sel(
                lat=lat, 
                lon=lon360, 
                time=event_date,
                method='nearest'
            )
            
            value = float(point_data.values)
            
            if np.isnan(value):
                print(f"NaN value for {scenario} at ({lat}, {lon}) on {event_date}")
                return {}
            
            return {f'{scenario}_precipitation': value}
            
        except Exception as e:
            print(f"Error extracting CMIP data for {scenario}: {e}")
            return {}

    def create_future_scenarios(self, df: pd.DataFrame,
                              years: List[int] = None,
                              lat_col: str = 'Latitude',
                              lon_col: str = 'Longitude',
                              date_col: str = 'event_date') -> Dict[str, pd.DataFrame]:
        """
        Create future scenario datasets for each CMIP6 scenario.
        
        Args:
            df: Input DataFrame
            years: List of years to create scenarios for
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            date_col: Name of date column
            
        Returns:
            Dictionary of DataFrames for each scenario
        """
        if years is None:
            years = list(range(2025, 2101, 5))  # Every 5 years from 2025-2100
        
        print(f"Creating future scenarios for years: {years}")
        
        scenario_datasets = {}
        
        for scenario in self.scenarios:
            print(f"Processing {scenario} scenario...")
            
            scenario_data = []
            
            for year in years:
                # Create dataset for this year
                year_df = df.copy()
                
                # Update event dates to future year
                year_df[date_col] = year_df[date_col].apply(
                    lambda x: x.replace(year=year) if pd.notna(x) else x
                )
                
                # Add CMIP6 data for each location
                cmip_features = []
                
                for idx, row in year_df.iterrows():
                    lat = row[lat_col]
                    lon = row[lon_col]
                    event_date = row[date_col]
                    
                    if pd.isna(lat) or pd.isna(lon) or pd.isna(event_date):
                        cmip_features.append({})
                        continue
                    
                    # Extract CMIP6 data
                    cmip_data = self.extract_cmip_data_for_location(
                        lat, lon, event_date, scenario
                    )
                    
                    cmip_features.append(cmip_data)
                
                # Add CMIP6 features to DataFrame
                for feature_name in cmip_features[0].keys():
                    values = [f.get(feature_name, np.nan) for f in cmip_features]
                    year_df[feature_name] = values
                
                # Add year and scenario information
                year_df['forecast_year'] = year
                year_df['scenario'] = scenario
                
                scenario_data.append(year_df)
            
            # Combine all years for this scenario
            if scenario_data:
                scenario_datasets[scenario] = pd.concat(scenario_data, ignore_index=True)
                print(f"Created {scenario} dataset with {len(scenario_datasets[scenario])} rows")
        
        return scenario_datasets

    def predict_stability_changes(self, df: pd.DataFrame,
                                scenario_datasets: Dict[str, pd.DataFrame],
                                stable_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict stability changes for future scenarios.
        
        Args:
            df: Original DataFrame
            scenario_datasets: Dictionary of future scenario datasets
            stable_threshold: Threshold for defining stability
            
        Returns:
            Dictionary with stability change predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_stability_model first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Train model first.")
        
        print("Predicting stability changes for future scenarios...")
        
        results = {}
        
        # Get original predictions
        original_predictions = self.model.predict(df[self.feature_columns].fillna(0))
        original_stable = original_predictions >= stable_threshold
        
        for scenario, scenario_df in scenario_datasets.items():
            print(f"Analyzing {scenario} scenario...")
            
            try:
                # Prepare features for prediction
                available_features = [col for col in self.feature_columns if col in scenario_df.columns]
                missing_features = [col for col in self.feature_columns if col not in scenario_df.columns]
                
                if missing_features:
                    print(f"Warning: Missing features for {scenario}: {missing_features}")
                
                # Fill missing features with zeros
                X_future = scenario_df[available_features].copy()
                for col in missing_features:
                    X_future[col] = 0
                
                # Reorder columns to match training data
                X_future = X_future[self.feature_columns]
                
                # Make predictions
                future_predictions = self.model.predict(X_future.fillna(0))
                future_stable = future_predictions >= stable_threshold
                
                # Calculate stability changes
                stability_changes = {
                    'stable_to_stable': ((original_stable) & (future_stable)).sum(),
                    'stable_to_unstable': ((original_stable) & (~future_stable)).sum(),
                    'unstable_to_stable': ((~original_stable) & (future_stable)).sum(),
                    'unstable_to_unstable': ((~original_stable) & (~future_stable)).sum(),
                    'total_points': len(original_stable),
                    'stability_rate': future_stable.mean(),
                    'stability_change': future_stable.mean() - original_stable.mean()
                }
                
                results[scenario] = {
                    'stability_changes': stability_changes,
                    'predictions': future_predictions,
                    'dataset': scenario_df
                }
                
                print(f"  {scenario} stability rate: {stability_changes['stability_rate']:.4f}")
                print(f"  {scenario} stability change: {stability_changes['stability_change']:+.4f}")
                
            except Exception as e:
                print(f"Error predicting stability for {scenario}: {e}")
                results[scenario] = None
        
        return results

    def generate_stability_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate a comprehensive stability report.
        
        Args:
            results: Results from predict_stability_changes
            
        Returns:
            DataFrame with stability report
        """
        report_data = []
        
        for scenario, result in results.items():
            if result is None:
                continue
            
            changes = result['stability_changes']
            
            report_data.append({
                'scenario': scenario,
                'total_points': changes['total_points'],
                'stable_to_stable': changes['stable_to_stable'],
                'stable_to_unstable': changes['stable_to_unstable'],
                'unstable_to_stable': changes['unstable_to_stable'],
                'unstable_to_unstable': changes['unstable_to_unstable'],
                'stability_rate': changes['stability_rate'],
                'stability_change': changes['stability_change']
            })
        
        report_df = pd.DataFrame(report_data)
        
        if not report_df.empty:
            print("\nStability Change Report:")
            print("="*80)
            print(report_df.to_string(index=False))
        
        return report_df

    def save_predictions(self, results: Dict[str, Any], 
                        output_dir: str = "output") -> None:
        """
        Save prediction results to files.
        
        Args:
            results: Results from predict_stability_changes
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed predictions for each scenario
        for scenario, result in results.items():
            if result is None:
                continue
            
            # Save dataset with predictions
            dataset = result['dataset'].copy()
            dataset['predicted_stability'] = result['predictions']
            
            output_file = output_path / f"stability_predictions_{scenario}.csv"
            dataset.to_csv(output_file, index=False)
            print(f"Saved {scenario} predictions to {output_file}")
        
        # Save summary report
        report_df = self.generate_stability_report(results)
        if not report_df.empty:
            report_file = output_path / "stability_summary_report.csv"
            report_df.to_csv(report_file, index=False)
            print(f"Saved stability summary to {report_file}")


def run_cmip_stability_analysis(df: pd.DataFrame,
                               cmip_data: Dict[str, xr.DataArray],
                               feature_columns: List[str],
                               target_column: str = 'Stability',
                               output_dir: str = "output") -> Dict[str, Any]:
    """
    Run complete CMIP stability analysis pipeline.
    
    Args:
        df: Input DataFrame
        cmip_data: CMIP6 rolling precipitation data
        feature_columns: List of feature columns
        target_column: Target column name
        output_dir: Output directory
        
    Returns:
        Dictionary with analysis results
    """
    print("Starting CMIP stability analysis...")
    
    # Initialize predictor
    predictor = CMIPModelPredictor(cmip_data)
    
    # Train model
    training_results = predictor.train_stability_model(
        df, feature_columns, target_column
    )
    
    if not training_results:
        print("Model training failed")
        return {}
    
    # Create future scenarios
    scenario_datasets = predictor.create_future_scenarios(df)
    
    if not scenario_datasets:
        print("No future scenarios created")
        return training_results
    
    # Predict stability changes
    stability_results = predictor.predict_stability_changes(
        df, scenario_datasets
    )
    
    # Generate and save reports
    predictor.save_predictions(stability_results, output_dir)
    
    # Combine results
    complete_results = {
        'training': training_results,
        'stability_predictions': stability_results,
        'scenario_datasets': scenario_datasets
    }
    
    print("CMIP stability analysis completed successfully!")
    return complete_results


if __name__ == "__main__":
    # Example usage
    print("CMIP Model Predictor Example")
    print("="*50)
    
    # This would require actual data
    # Example of how to use the predictor:
    
    # predictor = CMIPModelPredictor(cmip_data)
    # 
    # # Train model
    # results = predictor.train_stability_model(df, feature_columns)
    # 
    # # Create future scenarios
    # scenarios = predictor.create_future_scenarios(df)
    # 
    # # Predict stability changes
    # stability_results = predictor.predict_stability_changes(df, scenarios)
    
    print("CMIP model predictor initialized successfully!")
    print("Use run_cmip_stability_analysis() to run the complete pipeline.") 