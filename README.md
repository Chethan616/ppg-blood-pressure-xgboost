# PPG-Based Blood Pressure Estimation (SBP and DBP)

This project implements a complete machine learning pipeline to estimate systolic blood pressure (SBP) and diastolic blood pressure (DBP) from photoplethysmography (PPG) signals using XGBoost.

The implementation follows the full course-project workflow:
1. Data acquisition from MATLAB files
2. Signal preprocessing
3. Signal segmentation
4. Feature extraction
5. Label extraction from ABP
6. Dataset preparation
7. Model training with hyperparameter tuning
8. Evaluation (MAE, RMSE, R2)
9. Visualization
10. Final outputs for reporting

## Dataset

Kaggle dataset:
https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset

Expected local placement:
- Put all `part_*.mat` files inside `data/raw/`

Dataset notes used in this pipeline:
- Sampling frequency: 125 Hz
- Channel layout in each record matrix:
  - Row 1: PPG
  - Row 2: ABP
  - Row 3: ECG (ignored)

## Project Structure

```text
BP_PPG/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── results/
│   ├── models/
│   └── plots/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── label_extraction.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils.py
├── config.yaml
├── main.py
└── requirements.txt
```

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Full Pipeline

```powershell
python main.py --config config.yaml
```

The script prints intermediate outputs at each stage, including:
- Number of files and records loaded
- Rejected records and segments
- Feature matrix shape and sample rows
- Split information
- Hyperparameter tuning progress
- Final metrics and sample predictions

## Methodology Mapping

### 1) Data Acquisition
- `src/data_loader.py`
- Uses `scipy.io.loadmat` to read `part_*.mat`
- Extracts PPG and ABP from each record

### 2) Signal Preprocessing
- `src/preprocessing.py`
- Handles missing/corrupted values
- PPG bandpass filtering: 0.5 to 8 Hz (Butterworth, zero-phase)
- Segment-level normalization (`zscore`, configurable)

### 3) Signal Segmentation
- `src/segmentation.py`
- Default 5-second windows at 125 Hz (625 samples)
- Default overlap 50%

### 4) Feature Extraction
- `src/feature_extraction.py`
- Time-domain: mean, std, pulse amplitude
- Statistical: variance, skewness, kurtosis
- Derivative: first and second derivative features
- Frequency-domain: FFT dominant frequency and spectral energy

### 5) Label Extraction
- `src/label_extraction.py`
- SBP = max ABP in window
- DBP = min ABP in window

### 6) Dataset Preparation
- `main.py`
- Creates feature matrix X and targets y_sbp, y_dbp
- 80:20 train/test split
- Group-aware split when group metadata is available

### 7) Model Training
- `src/model_training.py`
- Two independent `XGBRegressor` models (SBP and DBP)
- Grid-search tuning on:
  - `n_estimators`
  - `max_depth`
  - `learning_rate`

### 8) Evaluation
- `src/evaluation.py`
- Reports MAE, RMSE, and R2 for both SBP and DBP

### 9) Visualization
- `src/evaluation.py`
- Saves:
  - Actual vs Predicted scatter plots
  - Feature importance plots

### 10) Output Artifacts
Generated artifacts include:
- `results/metrics.json`
- `results/models/xgb_sbp_model.joblib`
- `results/models/xgb_dbp_model.joblib`
- `results/models/feature_scaler.joblib`
- `results/plots/actual_vs_predicted_sbp.png`
- `results/plots/actual_vs_predicted_dbp.png`
- `results/plots/feature_importance_sbp.png`
- `results/plots/feature_importance_dbp.png`
- `data/processed/sample_predictions.csv`

## Configuration

All major settings are centralized in `config.yaml`:
- Sampling frequency
- Filter settings
- Window length and overlap
- Label validity thresholds
- Split strategy
- Model hyperparameter grid

## Notes for Academic Submission

- The pipeline is modular and reproducible.
- Fixed random seed is used for repeatability.
- The report should document assumptions on inferred grouping (patient IDs are not explicit in this dataset split).
- Include dataset citation in your report.

## Citation Request

If this dataset is used in your project report, cite the original works listed on the Kaggle dataset page:
- M. Kachuee et al., ISCAS 2015
- M. Kachuee et al., IEEE TBME 2016
