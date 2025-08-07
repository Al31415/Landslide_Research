# Landslide Research Analysis 
This repository houses datasets from CMIP6 (CESM2), NOAA, SSURGO, and USGS. It benchmarks models for landslide prediction, identifies the most influential features, and uses the top-performing models to project how landslide risk will evolve as climate-driven changes in precipitation intensify.

## Package Structure

```
modular_feature_analysis/
├── data                     # Folder that hosts our data (.nc files are too large, can be downloaded from https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=overview)
├── data_collection          # Folder that host scripts to collect the Meteostat (Weather Station Data), USGS (DEM Data), SSURGO (Soil Property Data), and CMIP6 data (Climate Data)
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

## Quick Start

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

## Modules Overview

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

## Advanced Usage

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

## Weighted Mean Rank Methodology

To evaluate the importance of features given the variance of feature importances for different models, we use a weighted mean rank approach:

### Formula
**Weighted Mean Rank = Σ(model_weight × feature_rank) / Σ(model_weights)**

Feature Importance
| Rank | Feature                    |    Score | Selected ¹ | Data Source        |
| ---- | -------------------------- | -------: | ---------: | ------------------ |
| 1    | Slope (USGS DEM)           | **1.00** |          3 | DEM                |
| 2    | 30-day mean precipitation  |     0.98 |          5 | Meteostat `_prcp`  |
| 3    | 1-day max precipitation    |     0.75 |          3 | Meteostat `_prcp`  |
| 4    | 365-day mean precipitation |     0.73 |          4 | Meteostat `_prcp`  |
| 5    | 90-day precipitation       |     0.71 |          4 | CESM2 `_mean_flux` |
| 6    | Deepest soil horizon       |     0.69 |          5 | SSURGO             |
| 7    | Slope class                |     0.68 |          3 | SSURGO             |
| 8    | 90-day precipitation       |     0.45 |          3 | Meteostat `_prcp`  |


¹ Number of models (out of 6) in which the feature was selected when using SelectFromModel.

### Model Performance (Weighted F₁)
| Model                        |           Dataset A |           Dataset B |
| ---------------------------- | ------------------: | ------------------: |
| **Random Forest**            | **0.9127 ± 0.0106** | **0.9237 ± 0.0139** |
| Gradient Boosting            |     0.9081 ± 0.0173 |     0.9214 ± 0.0096 |
| AdaBoost                     |     0.8970 ± 0.0152 |     0.9181 ± 0.0099 |
| Logistic Regression          |     0.8065 ± 0.0129 |     0.8317 ± 0.0223 |
| Ridge Classifier             |     0.7876 ± 0.0173 |     0.8328 ± 0.0243 |
| Linear Discriminant Analysis |     0.7942 ± 0.0182 |     0.7754 ± 0.0320 |



The top performing model is the Random Forest with Gradient Boosting trailing not far behind. 

## CMIP Forecast Conclusions 
Performing inference on this model with CMIP forecasts gives us a trend where with a median annual increase in precipitation of approximately 40 percent, stable slopes with already boreline unstable characteristics are predicted to become unstable. All locations that flip in stability, flip due to increases in precipitation (increases by 12.6, 18.3, 18.6, 19.3, 20.9, 22.3, 22.8, 23.1, 24.7, 42.3, 48.4, 59.8, 61.5, 69.8, 72.0, 72.5, 79.4, 83.9, 133.8 percent), except for one. At  39.3766, -87.1189, we see that a decrease in precipitation actually causes the location to flip. This feature has a high slope, high bulk density, and low saturated hydraulic conductivity. From this information, one can infer that the soil may be a dense clay that under low long term precipitation conditions forms desiccation cracks that lead to instability when precipitation does occur. 
## Output Files

The analysis generates several output files in the `analysis_results/` folder:

- `no_nans_ranking.csv`: Feature rankings for dataset with no NaNs (dataset A)
- `optimal_ranking.csv`: Feature rankings for optimal dataset (dataset with more features, dataset B)
- `dataset_a_optimized.csv`: Optimized dataset A
- `dataset_b_optimized.csv`: Optimized dataset B
- `gb_a_importances.csv`: GradientBoosting importances for dataset A
- `gb_b_importances.csv`: GradientBoosting importances for dataset B
- `rf_a_importances.csv`: RandomForest importances for dataset A
- `rf_b_importances.csv`: RandomForest importances for dataset B

##  Supported Models

The package evaluates multiple machine learning models but any scikit-learn model can be easily swapped in:

1. **Logistic Regression** (`logistic`)
2. **Random Forest** (`rf`)
3. **Gradient Boosting** (`gb`)
4. **Ridge Classifier** (`ridge`)
5. **Linear Discriminant Analysis** (`lda`)
6. **AdaBoost** (`ada`)

##  Analysis Pipeline

The complete pipeline consists of 6 steps:

1. **Data Loading & Preparation**: Load data and create two datasets (One with more features and less rows (from deleting rows will null values), and the other with less features but more rows)
2. **Feature Selection**: Optimize datasets using SelectFromModel
3. **Basic Model Training**: Train GB and RF models
4. **Comprehensive Evaluation**: Evaluate all models with cross-validation
5. **Feature Ranking**: Apply weighted mean rank methodology
6. **Visualization**: Create comprehensive plots (optional)

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0






