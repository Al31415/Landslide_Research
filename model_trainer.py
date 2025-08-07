"""
Model Training Module
Handles training different models and extracting feature importances.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Tuple, Any


def get_model_importances_gb(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Train GradientBoosting model and get feature importances.
    
    Args:
        df (pd.DataFrame): Input dataset
        dataset_name (str): Name of the dataset for logging
        
    Returns:
        pd.DataFrame: Feature importances DataFrame
    """
    # Train gradient boosting model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Fit model
    gb.fit(df.drop('Stability', axis=1), df['Stability'])
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': df.drop('Stability', axis=1).columns,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"GradientBoosting Feature importances for {dataset_name}:")
    print(importances)
    
    return importances


def get_model_importances_rf(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Train RandomForest model and get feature importances.
    
    Args:
        df (pd.DataFrame): Input dataset
        dataset_name (str): Name of the dataset for logging
        
    Returns:
        pd.DataFrame: Feature importances DataFrame
    """
    # Train random forest model
    rf = RandomForestClassifier(random_state=42)
    
    # Fit model
    rf.fit(df.drop('Stability', axis=1), df['Stability'])
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': df.drop('Stability', axis=1).columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Random Forest Feature importances for {dataset_name}:")
    print(importances)
    
    return importances


def clean_features_for_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean features for model evaluation by dropping unwanted columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Drop unwanted columns
    cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
    cols_to_drop.extend([col for col in df.columns if 'Precipitation' in col])
    cols_to_drop.extend(['event_date', 'Latitude', 'Longitude'])
    
    return df.drop(columns=cols_to_drop)


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, features: pd.Index, model_name: str) -> Tuple[float, float, pd.DataFrame]:
    """
    Evaluate a model and get feature importance rankings.
    
    Args:
        model: Sklearn model instance
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        features (pd.Index): Feature names
        model_name (str): Name of the model
        
    Returns:
        Tuple[float, float, pd.DataFrame]: CV score, CV std, importance DataFrame
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Select features using entire dataset
    selector = SelectFromModel(model, prefit=False)
    selector.fit(X, y)
    
    # Get selected features and their mask
    selected_mask = selector.get_support()
    all_selected_features = features[selected_mask]
    
    # Fit model on selected features
    X_selected = selector.transform(X)
    model.fit(X_selected, y)
    
    # Calculate cross-validation F1 scores
    cv_scores = cross_val_score(model, X_selected, y, scoring='f1_weighted', cv=cv)
    
    # Get feature importance and normalize to percentages
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    
    # Normalize importances to percentages
    importances = (importances / importances.sum()) * 100
    
    # Create importance DataFrame with ranks
    importance_df = pd.DataFrame({
        'feature': all_selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return np.mean(cv_scores), np.std(cv_scores), importance_df


def get_model_list() -> List[Tuple[str, Any]]:
    """
    Get list of models for evaluation.
    
    Returns:
        List[Tuple[str, Any]]: List of (name, model) tuples
    """
    return [
        ('logistic', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('ridge', RidgeClassifier(random_state=42)),
        ('lda', LinearDiscriminantAnalysis()),
        ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
    ]


def evaluate_all_models(df: pd.DataFrame, dataset_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all models on a dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        dataset_name (str): Name of the dataset
        
    Returns:
        Dict[str, Dict[str, Any]]: Results dictionary for each model
    """
    # Clean features
    df_clean = clean_features_for_evaluation(df)
    
    # Prepare data
    X = df_clean.drop('Stability', axis=1)
    y = df_clean['Stability']
    features = X.columns
    
    # Get models
    models = get_model_list()
    
    # Store results
    results = {}
    
    print(f"Results for {dataset_name}:")
    
    for name, model in models:
        try:
            cv_score, cv_std, importance_df = evaluate_model(model, X, y, features, name)
            results[name] = {
                'cv_score': cv_score,
                'cv_std': cv_std,
                'importance_df': importance_df
            }
            print(f"\n{name.upper()} CV F1-weighted Score: {cv_score:.4f} ± {cv_std:.4f}")
            print("Selected features and their percentage importance:")
            print(importance_df)
        except Exception as e:
            print(f"\nSkipping {name} due to error: {str(e)}")
    
    return results


def train_basic_models(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Train basic models (GB and RF) on both datasets.
    
    Args:
        df_a (pd.DataFrame): First dataset
        df_b (pd.DataFrame): Second dataset
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing importances for each model
    """
    results = {}
    
    # GradientBoosting importances
    results['gb_a'] = get_model_importances_gb(df_a, "Dataset A (No NaNs)")
    results['gb_b'] = get_model_importances_gb(df_b, "Dataset B (Optimal Threshold)")
    
    # RandomForest importances
    results['rf_a'] = get_model_importances_rf(df_a, "Dataset A (No NaNs)")
    results['rf_b'] = get_model_importances_rf(df_b, "Dataset B (Optimal Threshold)")
    
    return results


if __name__ == "__main__":
    # Test the model training functionality
    from data_loader import prepare_datasets
    from feature_selector import optimize_both_datasets
    
    # Load and prepare datasets
    datasets = prepare_datasets()
    
    if 'no_nans' in datasets and 'optimal' in datasets:
        # Optimize datasets
        optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])
        
        if 'df_a_optimal' in optimized and 'df_b_optimal' in optimized:
            # Train basic models
            basic_results = train_basic_models(optimized['df_a_optimal'], optimized['df_b_optimal'])
            print("\nBasic model training completed!")
            
            # Evaluate all models
            print("\nEvaluating all models...")
            all_results_a = evaluate_all_models(optimized['df_a_optimal'], "Dataset with No NaNs")
            all_results_b = evaluate_all_models(optimized['df_b_optimal'], "Optimal Dataset")
            
            print("\nModel evaluation completed!")
        else:
            print("Error: Optimized datasets not found.")
    else:
        print("Error: Required datasets not found.") 