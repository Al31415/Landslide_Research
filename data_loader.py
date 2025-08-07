"""
Data Loading Module
Handles loading and initial data preparation for feature analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_data(file_path: str = 'Corrected_Input_Data.csv') -> pd.DataFrame:
    """
    Load the main dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_no_nans_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset by removing all columns with any NaN values.
    
    Args:
        df (pd.DataFrame): Original dataset
        
    Returns:
        pd.DataFrame: Dataset with no NaN columns
    """
    df_no_nans = df.dropna(axis=1, how='any')
    print(f"Shape after removing all NaN columns: {df_no_nans.shape}")
    return df_no_nans


def find_optimal_threshold(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Find optimal threshold for NaN removal using weighted ratio.
    
    Args:
        df (pd.DataFrame): Original dataset
        
    Returns:
        Tuple[int, pd.DataFrame]: Best threshold and results DataFrame
    """
    results = []
    
    for threshold in range(1, len(df)):
        # Get columns with NaN count below threshold
        nan_counts = df.isna().sum()
        cols_to_keep = nan_counts[nan_counts < threshold].index
        
        if len(cols_to_keep) == 0:
            continue
            
        # Create temporary dataset with selected columns
        temp_df = df[cols_to_keep]
        
        # Remove rows with any remaining NaNs
        temp_df = temp_df.dropna(axis=0)
        
        # Calculate weighted ratio: (0.7 * feature_ratio + 0.3 * row_ratio)
        # Giving more weight to keeping features
        feature_ratio = len(cols_to_keep) / df.shape[1]
        row_ratio = len(temp_df) / df.shape[0]
        weighted_ratio = (0.7 * feature_ratio) + (0.3 * row_ratio)
        
        results.append({
            'threshold': threshold,
            'features_kept': len(cols_to_keep),
            'rows_kept': len(temp_df),
            'weighted_ratio': weighted_ratio
        })
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find threshold with best weighted ratio
    best_threshold = results_df.loc[results_df['weighted_ratio'].idxmax(), 'threshold']
    print(f"Best threshold found: {best_threshold}")
    
    return best_threshold, results_df


def create_optimal_dataset(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Create optimized dataset using the best threshold.
    
    Args:
        df (pd.DataFrame): Original dataset
        threshold (int): Optimal threshold for NaN removal
        
    Returns:
        pd.DataFrame: Optimized dataset
    """
    nan_counts = df.isna().sum()
    optimal_cols = nan_counts[nan_counts < threshold].index
    df_optimal = df[optimal_cols].dropna(axis=0)
    
    print(f"Optimal dataset shape: {df_optimal.shape}")
    print(f"Features kept: {len(optimal_cols)} ({len(optimal_cols)/df.shape[1]:.2%})")
    print(f"Rows kept: {len(df_optimal)} ({len(df_optimal)/df.shape[0]:.2%})")
    
    return df_optimal


def prepare_datasets(file_path: str = 'Corrected_Input_Data.csv') -> Dict[str, pd.DataFrame]:
    """
    Prepare both datasets (no NaNs and optimal) from the original data.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing both datasets
    """
    # Load data
    df = load_data(file_path)
    if df is None:
        return {}
    
    # Create no NaNs dataset
    df_no_nans = create_no_nans_dataset(df)
    
    # Find optimal threshold and create optimal dataset
    best_threshold, results_df = find_optimal_threshold(df)
    df_optimal = create_optimal_dataset(df, best_threshold)
    
    return {
        'original': df,
        'no_nans': df_no_nans,
        'optimal': df_optimal,
        'threshold_results': results_df,
        'best_threshold': best_threshold
    }


if __name__ == "__main__":
    # Test the data loading functionality
    datasets = prepare_datasets()
    print("\nDataset preparation completed!")
    print(f"Available datasets: {list(datasets.keys())}") 