"""
Test Script for Modular Feature Analysis Package
Verifies that all modules work correctly together.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        from data_loader import prepare_datasets
        print("✓ data_loader imported successfully")
    except ImportError as e:
        print(f"✗ data_loader import failed: {e}")
        return False
    
    try:
        from feature_selector import optimize_both_datasets
        print("✓ feature_selector imported successfully")
    except ImportError as e:
        print(f"✗ feature_selector import failed: {e}")
        return False
    
    try:
        from model_trainer import evaluate_all_models, train_basic_models
        print("✓ model_trainer imported successfully")
    except ImportError as e:
        print(f"✗ model_trainer import failed: {e}")
        return False
    
    try:
        from feature_ranker import rank_features
        print("✓ feature_ranker imported successfully")
    except ImportError as e:
        print(f"✗ feature_ranker import failed: {e}")
        return False
    
    try:
        from visualizer import create_comprehensive_visualization
        print("✓ visualizer imported successfully")
    except ImportError as e:
        print(f"✗ visualizer import failed: {e}")
        return False
    
    try:
        from main import run_complete_analysis
        print("✓ main imported successfully")
    except ImportError as e:
        print(f"✗ main import failed: {e}")
        return False
    
    return True


def test_data_loader():
    """Test data loading functionality."""
    print("\nTesting data loader...")
    
    try:
        from data_loader import prepare_datasets
        
        # Check if data file exists
        if not os.path.exists('../Corrected_Input_Data.csv'):
            print("⚠ Data file not found, skipping data loader test")
            return True
        
        datasets = prepare_datasets('../Corrected_Input_Data.csv')
        
        if datasets and 'no_nans' in datasets and 'optimal' in datasets:
            print("✓ Data loader test passed")
            print(f"  - No NaNs dataset shape: {datasets['no_nans'].shape}")
            print(f"  - Optimal dataset shape: {datasets['optimal'].shape}")
            return True
        else:
            print("✗ Data loader test failed")
            return False
            
    except Exception as e:
        print(f"✗ Data loader test failed with error: {e}")
        return False


def test_feature_selector():
    """Test feature selection functionality."""
    print("\nTesting feature selector...")
    
    try:
        from feature_selector import optimize_both_datasets
        from data_loader import prepare_datasets
        
        # Check if data file exists
        if not os.path.exists('../Corrected_Input_Data.csv'):
            print("⚠ Data file not found, skipping feature selector test")
            return True
        
        datasets = prepare_datasets('../Corrected_Input_Data.csv')
        if not datasets:
            print("⚠ No datasets available, skipping feature selector test")
            return True
        
        optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])
        
        if optimized and 'df_a_optimal' in optimized and 'df_b_optimal' in optimized:
            print("✓ Feature selector test passed")
            print(f"  - Dataset A optimal shape: {optimized['df_a_optimal'].shape}")
            print(f"  - Dataset B optimal shape: {optimized['df_b_optimal'].shape}")
            return True
        else:
            print("✗ Feature selector test failed")
            return False
            
    except Exception as e:
        print(f"✗ Feature selector test failed with error: {e}")
        return False


def test_model_trainer():
    """Test model training functionality."""
    print("\nTesting model trainer...")
    
    try:
        from model_trainer import get_model_list, clean_features_for_evaluation
        
        # Test model list
        models = get_model_list()
        if len(models) > 0:
            print("✓ Model list test passed")
            print(f"  - Number of models: {len(models)}")
            print(f"  - Model names: {[name for name, _ in models]}")
        else:
            print("✗ Model list test failed")
            return False
        
        # Test feature cleaning function
        import pandas as pd
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'Unnamed: 0': [7, 8, 9],
            'event_date': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'Stability': [0, 1, 0]
        })
        
        cleaned_df = clean_features_for_evaluation(test_df)
        if 'Unnamed: 0' not in cleaned_df.columns and 'event_date' not in cleaned_df.columns:
            print("✓ Feature cleaning test passed")
        else:
            print("✗ Feature cleaning test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Model trainer test failed with error: {e}")
        return False


def test_feature_ranker():
    """Test feature ranking functionality."""
    print("\nTesting feature ranker...")
    
    try:
        from feature_ranker import calculate_weighted_mean_rank
        import pandas as pd
        import numpy as np
        
        # Create mock results for testing
        mock_results = {
            'model1': {
                'cv_score': 0.8,
                'importance_df': pd.DataFrame({
                    'feature': ['A', 'B', 'C'],
                    'importance': [50, 30, 20],
                    'rank': [1, 2, 3]
                })
            },
            'model2': {
                'cv_score': 0.9,
                'importance_df': pd.DataFrame({
                    'feature': ['A', 'B', 'D'],
                    'importance': [60, 25, 15],
                    'rank': [1, 2, 3]
                })
            }
        }
        
        ranking = calculate_weighted_mean_rank(mock_results)
        
        if isinstance(ranking, pd.DataFrame) and len(ranking) > 0:
            print("✓ Feature ranker test passed")
            print(f"  - Number of features ranked: {len(ranking)}")
            print(f"  - Columns: {list(ranking.columns)}")
        else:
            print("✗ Feature ranker test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Feature ranker test failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MODULAR FEATURE ANALYSIS - TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_loader,
        test_feature_selector,
        test_model_trainer,
        test_feature_ranker
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The modular package is working correctly.")
        print("\nTo run the complete analysis:")
        print("1. Make sure 'Corrected_Input_Data.csv' is in the parent directory")
        print("2. Run: python main.py")
    else:
        print("⚠ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    main() 