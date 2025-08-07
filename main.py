"""
Main Orchestration Script
Runs the complete feature analysis pipeline using all modular components.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import time

# Import all modules
from data_loader import prepare_datasets
from feature_selector import optimize_both_datasets
from model_trainer import train_basic_models, evaluate_all_models
from feature_ranker import rank_features
from visualizer import create_comprehensive_visualization


def run_complete_analysis(data_file: str = 'Corrected_Input_Data.csv', 
                         create_visualizations: bool = True) -> Dict[str, Any]:
    """
    Run the complete feature analysis pipeline.
    
    Args:
        data_file: Path to the input data file
        create_visualizations: Whether to create visualizations
        
    Returns:
        Dict[str, Any]: Complete analysis results
    """
    print("="*80)
    print("MODULAR FEATURE ANALYSIS PIPELINE")
    print("="*80)
    
    start_time = time.time()
    results = {}
    
    # Step 1: Data Loading and Preparation
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*50)
    
    datasets = prepare_datasets(data_file)
    if not datasets:
        print("Error: Failed to load datasets. Exiting.")
        return {}
    
    results['datasets'] = datasets
    print(f"✓ Datasets prepared successfully")
    print(f"  - Original shape: {datasets['original'].shape}")
    print(f"  - No NaNs shape: {datasets['no_nans'].shape}")
    print(f"  - Optimal shape: {datasets['optimal'].shape}")
    
    # Step 2: Feature Selection
    print("\n" + "="*50)
    print("STEP 2: FEATURE SELECTION")
    print("="*50)
    
    optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])
    if not optimized or 'df_a_optimal' not in optimized or 'df_b_optimal' not in optimized:
        print("Error: Failed to optimize datasets. Exiting.")
        return {}
    
    results['optimized'] = optimized
    print(f"✓ Datasets optimized successfully")
    print(f"  - Dataset A optimal shape: {optimized['df_a_optimal'].shape}")
    print(f"  - Dataset B optimal shape: {optimized['df_b_optimal'].shape}")
    if 'common_features' in optimized:
        print(f"  - Common features: {len(optimized['common_features'])}")
    
    # Step 3: Basic Model Training
    print("\n" + "="*50)
    print("STEP 3: BASIC MODEL TRAINING")
    print("="*50)
    
    basic_results = train_basic_models(optimized['df_a_optimal'], optimized['df_b_optimal'])
    results['basic_results'] = basic_results
    print(f"✓ Basic models trained successfully")
    print(f"  - GradientBoosting and RandomForest models completed")
    
    # Step 4: Comprehensive Model Evaluation
    print("\n" + "="*50)
    print("STEP 4: COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    
    print("Evaluating models for dataset with no NaNs...")
    results_no_nans = evaluate_all_models(optimized['df_a_optimal'], "Dataset with No NaNs")
    
    print("Evaluating models for optimal dataset...")
    results_optimal = evaluate_all_models(optimized['df_b_optimal'], "Optimal Dataset")
    
    results['model_evaluation'] = {
        'no_nans': results_no_nans,
        'optimal': results_optimal
    }
    print(f"✓ All models evaluated successfully")
    print(f"  - Models evaluated: {len(results_no_nans)} for each dataset")
    
    # Step 5: Feature Ranking
    print("\n" + "="*50)
    print("STEP 5: FEATURE RANKING")
    print("="*50)
    
    rankings = rank_features(results_no_nans, results_optimal)
    results['rankings'] = rankings
    print(f"✓ Feature ranking completed successfully")
    print(f"  - Features ranked for both datasets")
    
    # Step 6: Visualization (Optional)
    if create_visualizations:
        print("\n" + "="*50)
        print("STEP 6: VISUALIZATION")
        print("="*50)
        
        try:
            create_comprehensive_visualization(
                optimized['df_a_optimal'], 
                optimized['df_b_optimal'], 
                rankings
            )
            print(f"✓ Visualizations created successfully")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results available in returned dictionary")
    
    return results


def save_results(results: Dict[str, Any], output_dir: str = "analysis_results") -> None:
    """
    Save analysis results to files.
    
    Args:
        results: Analysis results dictionary
        output_dir: Output directory for saving results
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save rankings
    if 'rankings' in results:
        results['rankings']['no_nans_ranking'].to_csv(
            f"{output_dir}/no_nans_ranking.csv", index=False
        )
        results['rankings']['optimal_ranking'].to_csv(
            f"{output_dir}/optimal_ranking.csv", index=False
        )
        print(f"✓ Rankings saved to {output_dir}/")
    
    # Save optimized datasets
    if 'optimized' in results:
        results['optimized']['df_a_optimal'].to_csv(
            f"{output_dir}/dataset_a_optimized.csv", index=False
        )
        results['optimized']['df_b_optimal'].to_csv(
            f"{output_dir}/dataset_b_optimized.csv", index=False
        )
        print(f"✓ Optimized datasets saved to {output_dir}/")
    
    # Save basic model results
    if 'basic_results' in results:
        for name, df in results['basic_results'].items():
            df.to_csv(f"{output_dir}/{name}_importances.csv", index=False)
        print(f"✓ Basic model results saved to {output_dir}/")


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the analysis results.
    
    Args:
        results: Analysis results dictionary
    """
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    if 'rankings' in results:
        print("\nTOP 5 FEATURES - DATASET WITH NO NANS:")
        top_no_nans = results['rankings']['no_nans_ranking'].head(5)
        for i, row in top_no_nans.iterrows():
            print(f"  {i+1}. {row['feature']} (rank: {row['weighted_mean_rank']:.4f})")
        
        print("\nTOP 5 FEATURES - OPTIMAL DATASET:")
        top_optimal = results['rankings']['optimal_ranking'].head(5)
        for i, row in top_optimal.iterrows():
            print(f"  {i+1}. {row['feature']} (rank: {row['weighted_mean_rank']:.4f})")
    
    if 'model_evaluation' in results:
        print("\nMODEL PERFORMANCE SUMMARY:")
        for dataset_name, model_results in results['model_evaluation'].items():
            print(f"\n{dataset_name.upper()}:")
            for model_name, model_data in model_results.items():
                cv_score = model_data['cv_score']
                cv_std = model_data['cv_std']
                print(f"  {model_name}: {cv_score:.4f} ± {cv_std:.4f}")


if __name__ == "__main__":
    # Run the complete analysis
    print("Starting Modular Feature Analysis...")
    
    # Run analysis
    results = run_complete_analysis(
        data_file='Corrected_Input_Data.csv',
        create_visualizations=True
    )
    
    if results:
        # Save results
        save_results(results)
        
        # Print summary
        print_summary(results)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("Check the 'analysis_results' folder for saved files.")
    else:
        print("Analysis failed. Please check the error messages above.") 