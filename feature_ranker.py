"""
Feature Ranking Module
Implements weighted mean rank methodology for feature importance aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def aggregate_weighted_ranks(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate feature ranks with performance weights using weighted mean rank.
    
    Args:
        results_dict: Dictionary containing model results
            {
                'logistic': {'cv_score': .., 'importance_df': ..},
                ...
            }
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            feature | agg_rank | consensus_score | sel_freq
    """
    # -- a. model weights ------------------------------------------------
    cv_scores = np.array([v['cv_score'] for v in results_dict.values()])
    perf_w = np.exp(cv_scores * 10)                 #   w_j  ∝  exp(β·perf)
    perf_w /= perf_w.sum()                           #   Σ w_j = 1

    models = list(results_dict.keys())
    n_models = len(models)

    print("\nModel weights  w_j  (Eq 1, β = 10):")
    for m, w, p in zip(models, perf_w, cv_scores):
        print(f"{m:10s}  w = {w:6.3f}   (CV = {p:.3f})")

    # -- b. collect individual ranks into a matrix ----------------------
    all_feats = sorted({f
                        for v in results_dict.values()
                        for f in v['importance_df']['feature']})

    ranks_mat = np.full((len(all_feats), n_models), fill_value=np.nan)
    for j, m in enumerate(models):
        df_imp = results_dict[m]['importance_df']
        for _, row in df_imp.iterrows():
            i = all_feats.index(row['feature'])
            ranks_mat[i, j] = row['rank']          # r_ij

    # treat "missing in model" as worst rank = max_rank + 1
    max_rank = np.nanmax(ranks_mat) + 1
    ranks_mat[np.isnan(ranks_mat)] = max_rank

    # -- c. weighted mean rank  R_i = Σ w_j · r_ij -----------------------
    agg_rank = (ranks_mat * perf_w).sum(axis=1)

    # -- d. rescale to [0,1] as a consensus importance score -------------
    score = 1.0 - (agg_rank - agg_rank.min()) / (agg_rank.max() - agg_rank.min())

    # -- e. selection frequency (how often feature was seen) -------------
    sel_freq = (ranks_mat != max_rank).sum(axis=1) / n_models

    return (pd.DataFrame({
                'feature': all_feats,
                'agg_rank': agg_rank,
                'consensus_score': score,
                'selection_freq': sel_freq
            })
            .sort_values('consensus_score', ascending=False)
            .reset_index(drop=True))


def calculate_weighted_mean_rank(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate weighted mean rank for features using CV scores as weights.
    
    Formula: Weighted Mean Rank = Σ(model_weight * feature_rank) / Σ(model_weights)
    where model_weight = CV_score for that model
    
    Args:
        results: Dictionary containing model results
        
    Returns:
        pd.DataFrame: DataFrame with weighted mean ranks
    """
    # Get all unique features
    all_features = set()
    for name in results.keys():
        all_features.update(results[name]['importance_df']['feature'])
    
    # Get CV scores as weights
    cv_scores = np.array([results[name]['cv_score'] for name in results.keys()])
    weights = cv_scores / cv_scores.sum()  # Normalize weights to sum to 1
    
    # Print model weights
    print("\nModel weights based on CV scores:")
    print("Formula for model weights: CV_score / sum(CV_scores)")
    for (name, weight) in zip(results.keys(), weights):
        print(f"{name}: {weight:.4f} (CV score: {cv_scores[list(results.keys()).index(name)]:.4f})")
    
    # Calculate weighted mean rank for each feature
    weighted_mean_ranks = {}
    print("\nFormula for weighted mean rank calculation:")
    print("For each feature:")
    print("weighted_mean_rank = Σ(model_weight * feature_rank) / Σ(model_weights)")
    print("where:")
    print("- model_weight is the normalized CV score")
    print("- feature_rank is the position of the feature in that model's ranking")
    print("- Lower rank values indicate better performance")
    
    for feature in all_features:
        weighted_rank_sum = 0
        total_weight = 0
        print(f"\nCalculation for feature '{feature}':")
        
        for (name, weight) in zip(results.keys(), weights):
            imp_df = results[name]['importance_df']
            if feature in imp_df['feature'].values:
                rank = imp_df[imp_df['feature'] == feature]['rank'].values[0]
                contribution = weight * rank
                weighted_rank_sum += contribution
                total_weight += weight
                print(f"  {name}: {weight:.4f} * {rank} = {contribution:.4f}")
            else:
                # If feature not selected by this model, assign worst rank
                max_rank = len(imp_df) + 1
                contribution = weight * max_rank
                weighted_rank_sum += contribution
                total_weight += weight
                print(f"  {name}: {weight:.4f} * {max_rank} (not selected) = {contribution:.4f}")
        
        weighted_mean_rank = weighted_rank_sum / total_weight
        weighted_mean_ranks[feature] = weighted_mean_rank
        print(f"  Total weighted rank sum: {weighted_rank_sum:.4f}")
        print(f"  Total weight: {total_weight:.4f}")
        print(f"  Weighted mean rank: {weighted_mean_rank:.4f}")
    
    # Convert to DataFrame and sort by weighted mean rank (lower is better)
    result_df = pd.DataFrame({
        'feature': list(weighted_mean_ranks.keys()),
        'weighted_mean_rank': list(weighted_mean_ranks.values())
    }).sort_values('weighted_mean_rank', ascending=True)
    
    # Add final rank column
    result_df['final_rank'] = range(1, len(result_df) + 1)
    
    return result_df


def rank_features(results_no_nans: Dict[str, Dict[str, Any]], 
                 results_optimal: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Rank features for both datasets using weighted mean rank.
    
    Args:
        results_no_nans: Results for dataset with no NaNs
        results_optimal: Results for optimal dataset
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing ranking results
    """
    print("="*80)
    print("FEATURE RANKING USING WEIGHTED MEAN RANK")
    print("="*80)
    
    # Calculate weighted mean rank for both datasets
    final_rank_no_nans = calculate_weighted_mean_rank(results_no_nans)
    final_rank_optimal = calculate_weighted_mean_rank(results_optimal)
    
    print("\n" + "="*80)
    print("FINAL WEIGHTED MEAN RANK RESULTS")
    print("="*80)
    
    print("\nWeighted Mean Rank for Dataset with No NaNs:")
    print("(Lower rank values indicate better feature performance)")
    print(final_rank_no_nans)
    
    print("\nWeighted Mean Rank for Optimal Dataset:")
    print("(Lower rank values indicate better feature performance)")
    print(final_rank_optimal)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nDataset with No NaNs:")
    print(f"  Total features ranked: {len(final_rank_no_nans)}")
    print(f"  Best feature: {final_rank_no_nans.iloc[0]['feature']} (rank: {final_rank_no_nans.iloc[0]['weighted_mean_rank']:.4f})")
    print(f"  Worst feature: {final_rank_no_nans.iloc[-1]['feature']} (rank: {final_rank_no_nans.iloc[-1]['weighted_mean_rank']:.4f})")
    
    print(f"\nOptimal Dataset:")
    print(f"  Total features ranked: {len(final_rank_optimal)}")
    print(f"  Best feature: {final_rank_optimal.iloc[0]['feature']} (rank: {final_rank_optimal.iloc[0]['weighted_mean_rank']:.4f})")
    print(f"  Worst feature: {final_rank_optimal.iloc[-1]['feature']} (rank: {final_rank_optimal.iloc[-1]['weighted_mean_rank']:.4f})")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("• Lower weighted mean rank = Better feature performance")
    print("• Weights are based on model CV scores (higher CV score = higher weight)")
    print("• Features not selected by a model get assigned the worst rank for that model")
    print("• Final ranking considers both feature importance and model performance")
    
    return {
        'no_nans_ranking': final_rank_no_nans,
        'optimal_ranking': final_rank_optimal
    }


if __name__ == "__main__":
    # Test the feature ranking functionality
    from data_loader import prepare_datasets
    from feature_selector import optimize_both_datasets
    from model_trainer import evaluate_all_models
    
    # Load and prepare datasets
    datasets = prepare_datasets()
    
    if 'no_nans' in datasets and 'optimal' in datasets:
        # Optimize datasets
        optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])
        
        if 'df_a_optimal' in optimized and 'df_b_optimal' in optimized:
            # Evaluate all models
            print("Evaluating models for dataset with no NaNs...")
            results_no_nans = evaluate_all_models(optimized['df_a_optimal'], "Dataset with No NaNs")
            
            print("Evaluating models for optimal dataset...")
            results_optimal = evaluate_all_models(optimized['df_b_optimal'], "Optimal Dataset")
            
            # Rank features
            rankings = rank_features(results_no_nans, results_optimal)
            print("\nFeature ranking completed!")
        else:
            print("Error: Optimized datasets not found.")
    else:
        print("Error: Required datasets not found.") 