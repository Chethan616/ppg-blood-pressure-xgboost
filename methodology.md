# Methodology (Simple Explanation)

## 1) Project Goal
We want to estimate blood pressure from PPG signals without a cuff.

- Input signal: PPG
- Reference signal for labels: ABP
- Targets:
  - SBP (Systolic Blood Pressure)
  - DBP (Diastolic Blood Pressure)

This project uses two regression models (XGBoost):
- Model 1 predicts SBP
- Model 2 predicts DBP

---

## 2) Pipeline Overview and Related Files

### Step A: Data Acquisition
What we do:
- Load Kaggle MAT files (part_*.mat)
- Extract PPG and ABP from each record

Related files:
- src/data_loader.py
- main.py

Output of this step:
- A list of records with synchronized PPG and ABP arrays

### Step B: Signal Preprocessing
What we do:
- Clean missing/corrupted values
- Apply PPG bandpass filter (0.5 to 8 Hz)
- Keep ABP cleaned but not bandpass-filtered for label integrity
- Normalize each segment later (z-score by default)

Related files:
- src/preprocessing.py
- main.py
- config.yaml

Output of this step:
- Cleaner signals ready for segmentation

### Step C: Signal Segmentation
What we do:
- Split each signal into fixed windows
- Default settings:
  - Sampling rate = 125 Hz
  - Window = 5 seconds = 625 samples
  - Overlap = 50%

Related files:
- src/segmentation.py
- config.yaml

Output of this step:
- Many small synchronized PPG and ABP windows

### Step D: Feature Extraction
What we do for each PPG window:
- Time features:
  - Mean
  - Standard deviation
  - Pulse amplitude (max - min)
- Statistical features:
  - Variance
  - Skewness
  - Kurtosis
- Derivative features:
  - First derivative (velocity) summary
  - Second derivative (acceleration) summary
- Frequency features:
  - Dominant frequency from FFT
  - Spectral energy

Related files:
- src/feature_extraction.py
- main.py

Output of this step:
- Feature matrix X

### Step E: Label Extraction
What we do from each ABP window:
- SBP = max(ABP window)
- DBP = min(ABP window)

Related files:
- src/label_extraction.py
- src/segmentation.py

Output of this step:
- Target vectors y_sbp and y_dbp

### Step F: Dataset Preparation
What we do:
- Build X, y_sbp, y_dbp
- Split train and test (80:20)
- Apply group-aware split if groups are available
- Scale features using StandardScaler (fit on train only)

Related files:
- main.py
- src/utils.py
- data/splits/split_indices.json (generated)

Output of this step:
- Train/test datasets for modeling

### Step G: Model Training (Dual XGBoost)
What we do:
- Train one model for SBP and one model for DBP
- Tune basic hyperparameters:
  - n_estimators
  - max_depth
  - learning_rate

Related files:
- src/model_training.py
- main.py
- config.yaml

Output of this step:
- Trained SBP model and DBP model

### Step H: Evaluation and Visualization
What we do:
- Compute metrics:
  - MAE
  - RMSE
  - R2
- Plot:
  - Actual vs Predicted
  - Feature Importance

Related files:
- src/evaluation.py
- main.py
- results/metrics.json (generated)
- results/plots/* (generated)

Output of this step:
- Final performance and plots for report

---

## 3) 5-Fold Cross Validation (Simple)

5-Fold Cross Validation means:
- Split data into 5 equal parts

Example with 100 samples:
- Fold 1 -> 20 samples
- Fold 2 -> 20 samples
- Fold 3 -> 20 samples
- Fold 4 -> 20 samples
- Fold 5 -> 20 samples

How training happens:

Iteration 1
- Train -> Fold 2,3,4,5 (80 samples)
- Validation -> Fold 1 (20 samples)

Iteration 2
- Train -> Fold 1,3,4,5
- Validation -> Fold 2

Iteration 3
- Train -> Fold 1,2,4,5
- Validation -> Fold 3

Iteration 4
- Train -> Fold 1,2,3,5
- Validation -> Fold 4

Iteration 5
- Train -> Fold 1,2,3,4
- Validation -> Fold 5

You get 5 scores.
Then:
- Take average of the 5 scores
- That average is the final CV performance

Why this is useful:
- More reliable than one single split
- Reduces luck/bias from a random split

In this project:
- We still keep a final hold-out test set (80:20 split)
- 5-fold CV is used inside training for hyperparameter tuning

---

## 4) Matching with Project Requirements

1. PPG input + ABP labels
- Yes, exactly implemented
- PPG is model input
- SBP/DBP are extracted from ABP windows

2. XGBoost dual regressor
- Yes, exactly implemented
- One model for SBP
- One model for DBP

3. Feature extraction
- Yes, implemented
- Time + statistical + derivative + frequency features

4. Validation approach
- Yes, implemented
- Hold-out test split (80:20)
- Plus 5-fold CV on training set for tuning

5. Modular code structure
- Yes, implemented
- Separate files for each pipeline stage
- Easier to explain, debug, and present academically

Suggested project sentence:
"We followed a pipeline of preprocessing, segmentation, feature extraction, and XGBoost regression using PPG signals. We used dual regressors for SBP and DBP and applied 5-fold cross-validation for robust evaluation."

---

## 5) What is XGBoost? (Very Simple)

XGBoost = Extreme Gradient Boosting.

It is a machine learning algorithm used for prediction tasks, including regression.

Simple idea:
- It does not rely on one big model.
- It builds many small decision trees step by step.
- Each new tree tries to fix errors made by previous trees.

Easy example:

Step 1:
- Tree 1 gives a rough BP prediction

Step 2:
- Tree 2 focuses on mistakes from Tree 1

Step 3:
- Tree 3 fixes remaining errors

This continues for many trees.
Final prediction is the combined result of all trees.

Why it is strong:
- Handles non-linear patterns well
- Works well with tabular handcrafted features
- Usually gives good performance with careful tuning

---

## 6) Actual Run Summary in This Workspace

Executed command:
- python main.py --config config.yaml

Data used:
- data/raw/part_1.mat

Generated outputs:
- results/metrics.json
- results/models/xgb_sbp_model.joblib
- results/models/xgb_dbp_model.joblib
- results/models/feature_scaler.joblib
- results/plots/actual_vs_predicted_sbp.png
- results/plots/actual_vs_predicted_dbp.png
- results/plots/feature_importance_sbp.png
- results/plots/feature_importance_dbp.png
- data/processed/sample_predictions.csv

Test metrics from this run:
- SBP: MAE 15.73, RMSE 20.47, R2 0.0519
- DBP: MAE 7.26, RMSE 10.68, R2 0.0444

Note:
- These values are from one data part and one experiment configuration.
- Performance can improve with more dataset parts, feature engineering, and tuning.
