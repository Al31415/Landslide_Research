# Modular Feature Analysis Package

A comprehensive, modular Python package for feature analysis and selection using weighted mean rank methodology.

## 📁 Package Structure

```
modular_feature_analysis/
├── __init__.py              # Package initialization
├── data_loader.py           # Data loading and preparation
├── feature_selector.py      # Feature selection using SelectFromModel
├── model_trainer.py         # Model training and evaluation
├── feature_ranker.py        # Weighted mean rank feature ranking
├── visualizer.py            # Visualization and plotting
├── main.py                  # Main orchestration script
├── requirements.txt         # Package dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

1. Navigate to the package directory:
```bash
cd modular_feature_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run the complete analysis pipeline:
```bash
python main.py
```

This will:
- Load and prepare your datasets
- Perform feature selection
- Train and evaluate multiple models
- Rank features using weighted mean rank
- Create visualizations
- Save results to `analysis_results/` folder

## 📊 Modules Overview

### 1. Data Loader (`data_loader.py`)
- **Purpose**: Load and prepare datasets
- **Key Functions**:
  - `load_data()`: Load CSV data
  - `create_no_nans_dataset()`: Remove all NaN columns
  - `find_optimal_threshold()`: Find optimal NaN removal threshold
  - `prepare_datasets()`: Prepare both datasets

### 2. Feature Selector (`feature_selector.py`)
- **Purpose**: Feature selection using SelectFromModel
- **Key Functions**:
  - `clean_features()`: Remove unwanted columns
  - `optimize_dataset_selectfrommodel()`: Optimize dataset with SelectFromModel
  - `optimize_both_datasets()`: Optimize both datasets

### 3. Model Trainer (`model_trainer.py`)
- **Purpose**: Train and evaluate multiple models
- **Key Functions**:
  - `evaluate_model()`: Evaluate individual model
  - `evaluate_all_models()`: Evaluate all models on dataset
  - `train_basic_models()`: Train basic models (GB, RF)

### 4. Feature Ranker (`feature_ranker.py`)
- **Purpose**: Implement weighted mean rank methodology
- **Key Functions**:
  - `calculate_weighted_mean_rank()`: Calculate weighted mean ranks
  - `rank_features()`: Rank features for both datasets
  - `aggregate_weighted_ranks()`: Alternative ranking method

### 5. Visualizer (`visualizer.py`)
- **Purpose**: Create comprehensive visualizations
- **Key Functions**:
  - `plot_feature_distributions()`: Feature distribution plots
  - `plot_drop_importance()`: Drop importance analysis
  - `plot_ranking_comparison()`: Ranking comparison plots

### 6. Main Orchestrator (`main.py`)
- **Purpose**: Run complete analysis pipeline
- **Key Functions**:
  - `run_complete_analysis()`: Run full pipeline
  - `save_results()`: Save results to files
  - `print_summary()`: Print analysis summary

## 🔧 Advanced Usage

### Running Individual Modules

You can run individual modules for specific analysis:

```python
# Data preparation only
from data_loader import prepare_datasets
datasets = prepare_datasets('your_data.csv')

# Feature selection only
from feature_selector import optimize_both_datasets
optimized = optimize_both_datasets(datasets['no_nans'], datasets['optimal'])

# Model evaluation only
from model_trainer import evaluate_all_models
results = evaluate_all_models(optimized['df_a_optimal'], "Dataset A")

# Feature ranking only
from feature_ranker import rank_features
rankings = rank_features(results_no_nans, results_optimal)
```

### Custom Analysis

```python
from main import run_complete_analysis

# Run without visualizations
results = run_complete_analysis(
    data_file='your_data.csv',
    create_visualizations=False
)

# Access specific results
rankings = results['rankings']
model_eval = results['model_evaluation']
```

## 📈 Weighted Mean Rank Methodology

The package implements a sophisticated weighted mean rank approach:

### Formula
**Weighted Mean Rank = Σ(model_weight × feature_rank) / Σ(model_weights)**

Where:
- `model_weight = CV_score / sum(CV_scores)` (normalized CV score)
- `feature_rank` = position of feature in model's ranking (1 = best)
- **Lower rank values indicate better feature performance**

### Benefits
- ✅ More intuitive: Lower rank = better feature
- ✅ Balanced weighting: Uses actual model performance
- ✅ Handles missing features: Assigns worst rank if not selected
- ✅ Robust ranking: Considers all models equally

## 📋 Output Files

The analysis generates several output files in the `analysis_results/` folder:

- `no_nans_ranking.csv`: Feature rankings for dataset with no NaNs
- `optimal_ranking.csv`: Feature rankings for optimal dataset
- `dataset_a_optimized.csv`: Optimized dataset A
- `dataset_b_optimized.csv`: Optimized dataset B
- `gb_a_importances.csv`: GradientBoosting importances for dataset A
- `gb_b_importances.csv`: GradientBoosting importances for dataset B
- `rf_a_importances.csv`: RandomForest importances for dataset A
- `rf_b_importances.csv`: RandomForest importances for dataset B

## 🎯 Supported Models

The package evaluates multiple machine learning models:

1. **Logistic Regression** (`logistic`)
2. **Random Forest** (`rf`)
3. **Gradient Boosting** (`gb`)
4. **Ridge Classifier** (`ridge`)
5. **Linear Discriminant Analysis** (`lda`)
6. **AdaBoost** (`ada`)

## 🔍 Analysis Pipeline

The complete pipeline consists of 6 steps:

1. **Data Loading & Preparation**: Load data and create two datasets
2. **Feature Selection**: Optimize datasets using SelectFromModel
3. **Basic Model Training**: Train GB and RF models
4. **Comprehensive Evaluation**: Evaluate all models with cross-validation
5. **Feature Ranking**: Apply weighted mean rank methodology
6. **Visualization**: Create comprehensive plots (optional)

## 🛠️ Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## 📝 Example Output

```
================================================================================
MODULAR FEATURE ANALYSIS PIPELINE
================================================================================

==================================================
STEP 1: DATA LOADING AND PREPARATION
==================================================
✓ Datasets prepared successfully
  - Original shape: (1089, 56)
  - No NaNs shape: (1089, 39)
  - Optimal shape: (728, 50)

==================================================
STEP 2: FEATURE SELECTION
==================================================
✓ Datasets optimized successfully
  - Dataset A optimal shape: (1089, 7)
  - Dataset B optimal shape: (728, 9)
  - Common features: 4

...

TOP 5 FEATURES - DATASET WITH NO NANS:
  1. Slope From USGS Elevation Data (rank: 1.2345)
  2. Bulk Density (rank: 2.3456)
  3. Deepest Soil Horizon Layer (rank: 3.4567)
  4. avg_365_day_prcp_mean_flux (rank: 4.5678)
  5. pH (rank: 5.6789)
```

## 🤝 Contributing

To extend the package:

1. Add new functionality to appropriate modules
2. Update the main orchestrator if needed
3. Add tests for new functions
4. Update documentation

## 📄 License

This package is provided as-is for educational and research purposes. 