#!/usr/bin/env python3
"""
CMIP Analysis Workflow
Complete pipeline for CMIP6 data collection and stability prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_collection.cmip_data_collector import CMIPDataCollector, collect_cmip_data_for_dataset
from data_collection.cmip_model_predictor import CMIPModelPredictor, run_cmip_stability_analysis


def main():
    """Run the complete CMIP analysis workflow."""
    
    print("="*80)
    print("CMIP6 CLIMATE ANALYSIS WORKFLOW")
    print("="*80)
    
    # Configuration
    data_dir = "data"
    input_file = "data/Corrected_Input_Data.csv"
    output_dir = "cmip_output"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Step 1: Load input data
    print("\nStep 1: Loading input data...")
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        print("Please ensure Corrected_Input_Data.csv is in the data directory")
        return
    except Exception as e:
        print(f"Error loading input data: {e}")
        return
    
    # Step 2: Initialize CMIP data collector
    print("\nStep 2: Initializing CMIP data collector...")
    try:
        collector = CMIPDataCollector(data_dir)
        
        # Check for CMIP files
        files = collector.find_cmip_files()
        if not files:
            print("No CMIP6 files found in data directory")
            print("Expected files:")
            for scenario, pattern in collector.file_patterns.items():
                print(f"  {pattern}")
            return
        
        print(f"Found {len(files)} CMIP6 files")
        for scenario, file_path in files.items():
            print(f"  {scenario}: {Path(file_path).name}")
            
    except Exception as e:
        print(f"Error initializing CMIP collector: {e}")
        return
    
    # Step 3: Process CMIP6 scenarios
    print("\nStep 3: Processing CMIP6 scenarios...")
    try:
        rolling_data = collector.process_all_scenarios()
        
        if not rolling_data:
            print("No CMIP6 data processed successfully")
            return
        
        print(f"Successfully processed {len(rolling_data)} scenarios:")
        for scenario, data in rolling_data.items():
            print(f"  {scenario}: {data.time.dt.year.min()}-{data.time.dt.year.max()}")
            
    except Exception as e:
        print(f"Error processing CMIP6 scenarios: {e}")
        return
    
    # Step 4: Create forecast dataset
    print("\nStep 4: Creating forecast dataset...")
    try:
        forecast_df = collector.create_forecast_dataset(
            df, rolling_data, 'Latitude', 'Longitude', 'event_date'
        )
        
        # Save forecast dataset
        forecast_file = Path(output_dir) / "cmip_forecast_data.csv"
        forecast_df.to_csv(forecast_file, index=False)
        print(f"Forecast dataset saved to {forecast_file}")
        
    except Exception as e:
        print(f"Error creating forecast dataset: {e}")
        return
    
    # Step 5: Define features for model training
    print("\nStep 5: Preparing features for model training...")
    
    # Define precipitation features (adjust based on your data)
    precip_features = [
        'max_1_day_prcp', 'max_3_day_prcp', 'max_7_day_prcp', 'max_14_day_prcp',
        'avg_30_day_prcp', 'avg_60_day_prcp', 'avg_90_day_prcp', 'avg_365_day_prcp'
    ]
    
    # Add CMIP features
    cmip_features = []
    for scenario in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        cmip_features.append(f'{scenario}_event_precipitation')
    
    # Combine all features (adjust based on your actual columns)
    all_features = precip_features + cmip_features
    
    # Filter to features that exist in the dataset
    available_features = [f for f in all_features if f in forecast_df.columns]
    
    if not available_features:
        print("Warning: No expected features found in dataset")
        print("Available columns:")
        for col in forecast_df.columns:
            print(f"  {col}")
        return
    
    print(f"Using {len(available_features)} features for model training:")
    for feature in available_features:
        print(f"  {feature}")
    
    # Step 6: Run stability analysis
    print("\nStep 6: Running stability analysis...")
    try:
        # Check if stability column exists
        if 'Stability' not in forecast_df.columns:
            print("Warning: 'Stability' column not found in dataset")
            print("Creating dummy stability values for demonstration...")
            # Create dummy stability values (replace with actual logic)
            forecast_df['Stability'] = np.random.choice([0, 1], size=len(forecast_df))
        
        # Run complete analysis
        results = run_cmip_stability_analysis(
            forecast_df, 
            rolling_data, 
            available_features, 
            'Stability', 
            output_dir
        )
        
        if results:
            print("Stability analysis completed successfully!")
            
            # Print summary
            if 'stability_predictions' in results:
                print("\nStability Prediction Summary:")
                for scenario, result in results['stability_predictions'].items():
                    if result:
                        changes = result['stability_changes']
                        print(f"  {scenario}:")
                        print(f"    Stability rate: {changes['stability_rate']:.4f}")
                        print(f"    Stability change: {changes['stability_change']:+.4f}")
        else:
            print("Stability analysis failed")
            
    except Exception as e:
        print(f"Error in stability analysis: {e}")
        return
    
    # Step 7: Generate summary report
    print("\nStep 7: Generating summary report...")
    try:
        summary_file = Path(output_dir) / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("CMIP6 Climate Analysis Summary\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Input data: {len(df)} rows\n")
            f.write(f"CMIP6 scenarios processed: {len(rolling_data)}\n")
            f.write(f"Features used: {len(available_features)}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            f.write("Generated files:\n")
            for file_path in Path(output_dir).glob("*"):
                f.write(f"  {file_path.name}\n")
        
        print(f"Summary report saved to {summary_file}")
        
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    print("\n" + "="*80)
    print("CMIP6 ANALYSIS WORKFLOW COMPLETED")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 