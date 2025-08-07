"""
Example Usage Script
Demonstrates how to use the modular feature analysis package.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """Example of basic usage - running the complete pipeline."""
    print("="*60)
    print("EXAMPLE 1: BASIC USAGE - COMPLETE PIPELINE")
    print("="*60)
    
    try:
        from main import run_complete_analysis
        
        # Run the complete analysis
        results = run_complete_analysis(
            data_file='../Corrected_Input_Data.csv',
            create_visualizations=False  # Set to True if you want plots
        )
        
        if results:
            print("✓ Analysis completed successfully!")
            print(f"Results keys: {list(results.keys())}")
        else:
            print("✗ Analysis failed")
            
    except Exception as e:
        print(f"Error in basic usage example: {e}")


def example_step_by_step():
    """Example of step-by-step usage."""
    print("\n" + "="*60)
    print("EXAMPLE 2: STEP-BY-STEP USAGE")
    print("="*60)
    
    try:
        # Step 1: Load data
        from data_loader import prepare_datasets
        print("Step 1: Loading data...")
        datasets = prepare_datasets('../Corrected_Input_Data.csv')
        
        if not datasets:
            print("✗ Failed to load data")
            return
        
        print(f"✓ Data loaded - No NaNs shape: {datasets['no_nans'].shape}")
        
        # Step 2: Feature selection
        from feature_selector import optimize_both_datasets
        print("\nStep 2: Feature selection...")
        optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])
        
        if not optimized:
            print("✗ Failed to optimize datasets")
            return
        
        print(f"✓ Datasets optimized - A: {optimized['df_a_optimal'].shape}, B: {optimized['df_b_optimal'].shape}")
        
        # Step 3: Model evaluation
        from model_trainer import evaluate_all_models
        print("\nStep 3: Model evaluation...")
        results_no_nans = evaluate_all_models(optimized['df_a_optimal'], "Dataset with No NaNs")
        results_optimal = evaluate_all_models(optimized['df_b_optimal'], "Optimal Dataset")
        
        print(f"✓ Models evaluated - {len(results_no_nans)} models for each dataset")
        
        # Step 4: Feature ranking
        from feature_ranker import rank_features
        print("\nStep 4: Feature ranking...")
        rankings = rank_features(results_no_nans, results_optimal)
        
        print("✓ Features ranked successfully")
        print(f"Top 3 features (No NaNs):")
        for i, row in rankings['no_nans_ranking'].head(3).iterrows():
            print(f"  {i+1}. {row['feature']} (rank: {row['weighted_mean_rank']:.4f})")
        
        # Step 5: Save results
        from main import save_results
        print("\nStep 5: Saving results...")
        save_results({
            'rankings': rankings,
            'optimized': optimized
        }, 'example_results')
        
        print("✓ Results saved to 'example_results' folder")
        
    except Exception as e:
        print(f"Error in step-by-step example: {e}")


def example_custom_analysis():
    """Example of custom analysis using specific modules."""
    print("\n" + "="*60)
    print("EXAMPLE 3: CUSTOM ANALYSIS")
    print("="*60)
    
    try:
        # Load data
        from data_loader import prepare_datasets
        datasets = prepare_datasets('../Corrected_Input_Data.csv')
        
        if not datasets:
            print("✗ Failed to load data")
            return
        
        # Use only the no NaNs dataset
        df_no_nans = datasets['no_nans']
        print(f"Using dataset with shape: {df_no_nans.shape}")
        
        # Clean features manually
        from feature_selector import clean_features
        df_clean = clean_features(df_no_nans)
        print(f"After cleaning: {df_clean.shape}")
        
        # Train only Random Forest
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        X = df_clean.drop('Stability', axis=1)
        y = df_clean['Stability']
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(rf, X, y, scoring='f1_weighted', cv=cv)
        print(f"Random Forest CV F1 score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Get feature importances
        rf.fit(X, y)
        import pandas as pd
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 features by Random Forest importance:")
        for i, row in importances.head(5).iterrows():
            print(f"  {i+1}. {row['feature']} ({row['importance']:.4f})")
        
        print("✓ Custom analysis completed")
        
    except Exception as e:
        print(f"Error in custom analysis example: {e}")


def example_visualization_only():
    """Example of running only visualization with existing results."""
    print("\n" + "="*60)
    print("EXAMPLE 4: VISUALIZATION ONLY")
    print("="*60)
    
    try:
        # This example assumes you have existing results
        # In practice, you would load saved results or run analysis first
        
        print("Note: This example requires existing analysis results.")
        print("Run one of the previous examples first to generate results.")
        
        # Example of how to create visualizations with existing data
        from visualizer import plot_feature_importance_comparison
        import pandas as pd
        
        # Create mock data for demonstration
        mock_importances_a = pd.DataFrame({
            'feature': ['Feature A', 'Feature B', 'Feature C'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        mock_importances_b = pd.DataFrame({
            'feature': ['Feature A', 'Feature B', 'Feature D'],
            'importance': [0.6, 0.25, 0.15]
        })
        
        print("Creating mock visualization...")
        # Uncomment the line below to actually create plots
        # plot_feature_importance_comparison(mock_importances_a, mock_importances_b, "Dataset A", "Dataset B")
        
        print("✓ Visualization example completed (plots disabled for demo)")
        
    except Exception as e:
        print(f"Error in visualization example: {e}")


def main():
    """Run all examples."""
    print("MODULAR FEATURE ANALYSIS - USAGE EXAMPLES")
    print("="*60)
    
    # Check if data file exists
    if not os.path.exists('../Corrected_Input_Data.csv'):
        print("⚠ Warning: 'Corrected_Input_Data.csv' not found in parent directory")
        print("Some examples may not work without the data file.")
        print("Please ensure the data file is available before running examples.\n")
    
    examples = [
        example_basic_usage,
        example_step_by_step,
        example_custom_analysis,
        example_visualization_only
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"Example {i} failed: {e}")
        
        print("\n" + "-"*60 + "\n")
    
    print("="*60)
    print("USAGE EXAMPLES COMPLETED")
    print("="*60)
    print("For more information, see the README.md file.")


if __name__ == "__main__":
    import pandas as pd  # Import here for the custom analysis example
    main() 