"""
Feature Selection Module
Handles feature selection using SelectFromModel and other selection methods.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean features by dropping unwanted columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Drop unwanted columns
    cols_to_drop = ['event_date', 'Latitude', 'Longitude'] + \
                   [col for col in df.columns if 'Unnamed' in col] + \
                   [col for col in df.columns if col.endswith('_nc')]
    
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    print(f"Shape after dropping columns: {df_clean.shape}")
    
    return df_clean


def optimize_dataset_selectfrommodel(df: pd.DataFrame, name: str) -> Optional[pd.DataFrame]:
    """
    Optimize dataset using SelectFromModel with GradientBoostingClassifier.
    
    Args:
        df (pd.DataFrame): Input dataset
        name (str): Dataset name for logging
        
    Returns:
        Optional[pd.DataFrame]: Optimized dataset or None if failed
    """
    print(f"\nOptimizing dataset {name} with SelectFromModel:")
    
    # Clean features
    df_clean = clean_features(df)
    
    # Ensure Stability column exists for prediction
    if 'Stability' not in df_clean.columns:
        print(f"Warning: Stability column not found in dataset {name}")
        return None
        
    # Remove any remaining NaN values
    df_clean = df_clean.dropna(axis=0)
    
    # Split features and target
    X = df_clean.drop('Stability', axis=1)
    y = df_clean['Stability']
    
    # Initialize gradient boosting model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Create F1 scorer
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    # Get initial cross validation score using F1
    initial_score = np.mean(cross_val_score(gb, X, y, cv=5, scoring=f1_scorer))
    print(f"Initial F1 score: {initial_score:.4f}")
    
    # Feature selection using SelectFromModel
    selector = SelectFromModel(gb, prefit=False)
    selector.fit(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Create optimized dataset with selected features
    X_selected = X[selected_features]
    
    # Get new cross validation score using F1
    final_score = np.mean(cross_val_score(gb, X_selected, y, cv=5, scoring=f1_scorer))
    print(f"Final F1 score with selected features: {final_score:.4f}")
    
    print(f"Features kept: {len(selected_features)} ({len(selected_features)/len(X.columns):.2%})")
    print(f"Selected features: {selected_features}")
    print(f"Rows kept: {len(X_selected)} ({len(X_selected)/df.shape[0]:.2%})")
    
    return X_selected.join(y)


def get_common_features(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    """
    Get common features between two optimized datasets.
    
    Args:
        df_a (pd.DataFrame): First optimized dataset
        df_b (pd.DataFrame): Second optimized dataset
        
    Returns:
        List[str]: List of common features
    """
    common_features = set(df_a.columns) & set(df_b.columns)
    print(f"\nNumber of common features: {len(common_features)}")
    print("Common features:")
    print(sorted(list(common_features)))
    
    return sorted(list(common_features))


def optimize_both_datasets(df_no_nans: pd.DataFrame, df_optimal: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Optimize both datasets using SelectFromModel.
    
    Args:
        df_no_nans (pd.DataFrame): Dataset with no NaNs
        df_optimal (pd.DataFrame): Optimal dataset
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing optimized datasets
    """
    print("="*80)
    df_a_optimal = optimize_dataset_selectfrommodel(df_no_nans, "A (No NaNs)")
    print("="*80) 
    df_b_optimal = optimize_dataset_selectfrommodel(df_optimal, "B (Optimal Threshold)")
    print("="*80)
    
    optimized_datasets = {}
    
    if df_a_optimal is not None:
        optimized_datasets['df_a_optimal'] = df_a_optimal
        
    if df_b_optimal is not None:
        optimized_datasets['df_b_optimal'] = df_b_optimal
    
    # Get common features if both datasets are available
    if df_a_optimal is not None and df_b_optimal is not None:
        common_features = get_common_features(df_a_optimal, df_b_optimal)
        optimized_datasets['common_features'] = common_features
    
    return optimized_datasets


if __name__ == "__main__":
    # Test the feature selection functionality
    from data_loader import prepare_datasets
    
    # Load datasets
    datasets = prepare_datasets()
    
    if 'no_nans' in datasets and 'optimal' in datasets:
        # Optimize datasets
        optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])
        print("\nFeature selection completed!")
        print(f"Optimized datasets: {list(optimized.keys())}")
    else:
        print("Error: Required datasets not found.") 