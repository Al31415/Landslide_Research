"""
Visualization Module
Handles plotting and visual analysis of feature distributions and importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.ensemble import GradientBoostingClassifier


def plot_feature_distributions(df_a: pd.DataFrame, df_b: pd.DataFrame, 
                             top_features_a: List[str], top_features_b: List[str]) -> None:
    """
    Create comprehensive feature distribution plots.
    
    Args:
        df_a: First dataset
        df_b: Second dataset
        top_features_a: Top features for dataset A
        top_features_b: Top features for dataset B
    """
    # Create figure with 3 subplots for each dataset
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('Feature Distributions Analysis', fontsize=16)
    
    # Plot overall distributions
    sns.boxplot(data=df_a[top_features_a], ax=axes[0,0])
    axes[0,0].set_title('Dataset A - Overall Feature Distributions')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df_b[top_features_b], ax=axes[0,1])
    axes[0,1].set_title('Dataset B - Overall Feature Distributions')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot distributions by stability class
    stability_dfs_a = []
    for feature in top_features_a:
        temp_df = pd.DataFrame({
            'value': df_a[feature],
            'Stability': df_a['Stability'],
            'Feature': feature
        })
        stability_dfs_a.append(temp_df)
    stability_data_a = pd.concat(stability_dfs_a)
    
    sns.boxplot(data=stability_data_a, x='Feature', y='value', hue='Stability', ax=axes[1,0])
    axes[1,0].set_title('Dataset A - Feature Distributions by Stability Class')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    stability_dfs_b = []
    for feature in top_features_b:
        temp_df = pd.DataFrame({
            'value': df_b[feature],
            'Stability': df_b['Stability'],
            'Feature': feature
        })
        stability_dfs_b.append(temp_df)
    stability_data_b = pd.concat(stability_dfs_b)
    
    sns.boxplot(data=stability_data_b, x='Feature', y='value', hue='Stability', ax=axes[1,1])
    axes[1,1].set_title('Dataset B - Feature Distributions by Stability Class')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Plot median values by stability class
    median_by_stability_a = df_a.groupby('Stability')[top_features_a].median()
    sns.heatmap(median_by_stability_a, cmap='YlOrRd', annot=True, fmt='.2f', ax=axes[2,0])
    axes[2,0].set_title('Dataset A - Median Values by Stability Class')
    
    median_by_stability_b = df_b.groupby('Stability')[top_features_b].median()
    sns.heatmap(median_by_stability_b, cmap='YlOrRd', annot=True, fmt='.2f', ax=axes[2,1])
    axes[2,1].set_title('Dataset B - Median Values by Stability Class')
    
    plt.tight_layout()
    plt.show()


def calculate_drop_importance(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Calculate drop column importance for features.
    
    Args:
        df: Input dataset
        dataset_name: Name of the dataset
        
    Returns:
        pd.DataFrame: Drop importance DataFrame
    """
    features = df.drop('Stability', axis=1).columns
    importances_drop = []
    
    # Calculate base score
    base_score = GradientBoostingClassifier(random_state=42).fit(
        df.drop('Stability', axis=1), df['Stability']
    ).score(df.drop('Stability', axis=1), df['Stability'])
    
    # Calculate drop importance for each feature
    for feature in features:
        X_dropped = df.drop(['Stability', feature], axis=1)
        gb_temp = GradientBoostingClassifier(random_state=42)
        gb_temp.fit(X_dropped, df['Stability'])
        score = gb_temp.score(X_dropped, df['Stability'])
        importance = base_score - score
        importances_drop.append(importance)
    
    # Create DataFrame
    importances_drop_df = pd.DataFrame({
        'feature': features,
        'drop_importance': importances_drop
    }).sort_values('drop_importance', ascending=False)
    
    print(f"\nDrop column importances for {dataset_name}:")
    print(importances_drop_df)
    
    return importances_drop_df


def plot_drop_importance(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """
    Plot drop column importances for both datasets.
    
    Args:
        df_a: First dataset
        df_b: Second dataset
    """
    # Calculate drop importances
    importances_a_drop_df = calculate_drop_importance(df_a, "Dataset A (No NaNs)")
    importances_b_drop_df = calculate_drop_importance(df_b, "Dataset B (Optimal Threshold)")
    
    # Plot drop column importances
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.barh(importances_a_drop_df['feature'][:10], importances_a_drop_df['drop_importance'][:10])
    plt.title('Top 10 Features by Drop Importance\nDataset A')
    plt.xlabel('Drop Importance')
    
    plt.subplot(1, 2, 2)
    plt.barh(importances_b_drop_df['feature'][:10], importances_b_drop_df['drop_importance'][:10])
    plt.title('Top 10 Features by Drop Importance\nDataset B')
    plt.xlabel('Drop Importance')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance_comparison(importances_a: pd.DataFrame, importances_b: pd.DataFrame,
                                     title_a: str, title_b: str) -> None:
    """
    Plot feature importance comparison between two datasets.
    
    Args:
        importances_a: Feature importances for dataset A
        importances_b: Feature importances for dataset B
        title_a: Title for dataset A
        title_b: Title for dataset B
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot for dataset A
    axes[0].barh(importances_a['feature'][:10], importances_a['importance'][:10])
    axes[0].set_title(f'Top 10 Features - {title_a}')
    axes[0].set_xlabel('Importance')
    
    # Plot for dataset B
    axes[1].barh(importances_b['feature'][:10], importances_b['importance'][:10])
    axes[1].set_title(f'Top 10 Features - {title_b}')
    axes[1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.show()


def plot_ranking_comparison(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame,
                          title_a: str, title_b: str) -> None:
    """
    Plot feature ranking comparison between two datasets.
    
    Args:
        ranking_a: Feature ranking for dataset A
        ranking_b: Feature ranking for dataset B
        title_a: Title for dataset A
        title_b: Title for dataset B
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot for dataset A (lower rank = better)
    top_features_a = ranking_a.head(10)
    axes[0].barh(range(len(top_features_a)), top_features_a['weighted_mean_rank'])
    axes[0].set_yticks(range(len(top_features_a)))
    axes[0].set_yticklabels(top_features_a['feature'], fontsize=8)
    axes[0].set_title(f'Top 10 Features - {title_a}\n(Lower rank = better)')
    axes[0].set_xlabel('Weighted Mean Rank')
    
    # Plot for dataset B (lower rank = better)
    top_features_b = ranking_b.head(10)
    axes[1].barh(range(len(top_features_b)), top_features_b['weighted_mean_rank'])
    axes[1].set_yticks(range(len(top_features_b)))
    axes[1].set_yticklabels(top_features_b['feature'], fontsize=8)
    axes[1].set_title(f'Top 10 Features - {title_b}\n(Lower rank = better)')
    axes[1].set_xlabel('Weighted Mean Rank')
    
    plt.tight_layout()
    plt.show()


def create_comprehensive_visualization(df_a: pd.DataFrame, df_b: pd.DataFrame,
                                     rankings: Dict[str, pd.DataFrame]) -> None:
    """
    Create comprehensive visualization suite for feature analysis.
    
    Args:
        df_a: First dataset
        df_b: Second dataset
        rankings: Dictionary containing ranking results
    """
    print("="*80)
    print("CREATING COMPREHENSIVE VISUALIZATION SUITE")
    print("="*80)
    
    # Get top features for visualization
    top_features_a = rankings['no_nans_ranking'].head(6)['feature'].tolist()
    top_features_b = rankings['optimal_ranking'].head(8)['feature'].tolist()
    
    # 1. Feature distributions
    print("\n1. Creating feature distribution plots...")
    plot_feature_distributions(df_a, df_b, top_features_a, top_features_b)
    
    # 2. Drop importance analysis
    print("\n2. Creating drop importance plots...")
    plot_drop_importance(df_a, df_b)
    
    # 3. Ranking comparison
    print("\n3. Creating ranking comparison plots...")
    plot_ranking_comparison(
        rankings['no_nans_ranking'], 
        rankings['optimal_ranking'],
        "Dataset with No NaNs", 
        "Optimal Dataset"
    )
    
    print("\nVisualization suite completed!")


if __name__ == "__main__":
    # Test the visualization functionality
    from data_loader import prepare_datasets
    from feature_selector import optimize_both_datasets
    from model_trainer import train_basic_models
    from feature_ranker import rank_features, evaluate_all_models
    
    # Load and prepare datasets
    datasets = prepare_datasets()
    
    if 'no_nans' in datasets and 'optimal' in datasets:
        # Optimize datasets
        optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])
        
        if 'df_a_optimal' in optimized and 'df_b_optimal' in optimized:
            # Train basic models and get importances
            basic_results = train_basic_models(optimized['df_a_optimal'], optimized['df_b_optimal'])
            
            # Evaluate all models
            results_no_nans = evaluate_all_models(optimized['df_a_optimal'], "Dataset with No NaNs")
            results_optimal = evaluate_all_models(optimized['df_b_optimal'], "Optimal Dataset")
            
            # Rank features
            rankings = rank_features(results_no_nans, results_optimal)
            
            # Create comprehensive visualization
            create_comprehensive_visualization(
                optimized['df_a_optimal'], 
                optimized['df_b_optimal'], 
                rankings
            )
        else:
            print("Error: Optimized datasets not found.")
    else:
        print("Error: Required datasets not found.") 