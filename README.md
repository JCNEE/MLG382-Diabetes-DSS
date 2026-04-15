# MLG382-Diabetes-DSS

Diabetes Decision Support System project that predicts diabetes stage, identifies the most important lifestyle and clinical drivers of risk, and groups patients into lifestyle-based segments.

## Current Pipeline Scope

This README covers the workflow from `src/prepare_data.py` to `src/train_models.py`.

The current pipeline does four things:
- Splits the raw dataset into train and test sets.
- Preprocesses the features and saves the encoders and scaler.
- Trains three supervised classification models for diabetes stage prediction.
- Trains a K-Means model for patient segmentation.

## Project Structure

- `data/Diabetes_and_LifeStyle_Dataset_.csv`: raw dataset.
- `src/prepare_data.py`: creates the train and test split and basic EDA plots.
- `src/preprocess_data.py`: removes leakage, encodes features, scales data, and saves processed files.
- `src/train_models.py`: balances the training data with SMOTENC, trains the models, evaluates them, and saves artifacts.
- `artifacts/`: stores plots, encoders, confusion matrices, and trained models.
- `data/`: stores the split data and processed modeling tables.

## Run Order

Run the scripts in this order:

```bash
python src/prepare_data.py
python src/preprocess_data.py
python src/train_models.py
```

If you are using the local virtual environment in this project, the Windows command is:

```powershell
".\venv\Scripts\python.exe" src/prepare_data.py
".\venv\Scripts\python.exe" src/preprocess_data.py
".\venv\Scripts\python.exe" src/train_models.py
```

## Step 1: Data Preparation

Script: `src/prepare_data.py`

Purpose:
- Loads the raw dataset from `data/Diabetes_and_LifeStyle_Dataset_.csv`.
- Splits the data into training and test sets using a stratified split on `diabetes_stage`.
- Saves the split datasets as `data/train.csv` and `data/test.csv`.
- Creates basic exploratory outputs for the full dataset.

What the script does:
- Uses `train_test_split(..., stratify=raw_data['diabetes_stage'])` so the class distribution is preserved across train and test.
- Saves the target distribution bar chart to `artifacts/target_distribution.png`.
- Saves the numeric correlation heatmap to `artifacts/correlation_heatmap.png`.

Files produced:
- `data/train.csv`
- `data/test.csv`
- `artifacts/target_distribution.png`
- `artifacts/correlation_heatmap.png`

## Step 2: Preprocessing

Script: `src/preprocess_data.py`

Purpose:
- Defines the target and feature groups.
- Removes data leakage.
- Encodes categorical data for the machine learning models.
- Scales the feature matrix for clustering.
- Saves the processed training and test tables plus reusable preprocessing artifacts.

### Feature Setup

The target is:
- `diabetes_stage`

The current leakage columns removed before modeling are:
- `diagnosed_diabetes`
- `diabetes_risk_score`

The current feature set contains 28 predictor columns:
- 19 numeric columns
- 3 binary columns
- 2 ordinal columns
- 4 nominal columns

### Encoding and Scaling Logic

The preprocessing script handles features as follows:
- Target encoding: `LabelEncoder` converts the diabetes stage labels into numeric class codes.
- Ordinal encoding: `education_level` and `income_level` are encoded with a fixed ordered category list.
- Nominal encoding: `gender`, `ethnicity`, `employment_status`, and `smoking_status` are label-encoded.
- Scaling: `StandardScaler` is applied to the processed feature matrix for K-Means.

### Why Two Feature Tables Are Saved

The script saves two versions of the feature matrix:
- Unscaled features: used for Decision Tree, Random Forest, XGBoost, and SHAP.
- Scaled features: used for K-Means.

Files produced:
- `data/X_train.csv`
- `data/X_test.csv`
- `data/y_train.csv`
- `data/y_test.csv`
- `data/X_train_scaled.csv`
- `data/X_test_scaled.csv`
- `artifacts/target_encoder.pkl`
- `artifacts/feature_encoders.pkl`
- `artifacts/scaler.pkl`

## Step 3: Model Training and Evaluation

Script: `src/train_models.py`

Purpose:
- Loads the processed data from the preprocessing step.
- Balances the training split with SMOTENC.
- Trains three classification models for diabetes stage prediction.
- Evaluates those models on the original test set.
- Compares the classifiers and selects the best one by Macro F1.
- Trains a K-Means model for patient segmentation.

### Why SMOTENC Is Used

The dataset is highly imbalanced across diabetes classes. `train_models.py` applies SMOTENC to the training split only so the classifiers do not simply learn to favor the majority classes.

Important details:
- SMOTENC is applied only to `X_train` and `y_train`.
- The test set stays untouched so evaluation still reflects real-world class imbalance.
- The code treats binary, ordinal, and nominal columns as categorical when creating synthetic minority samples.

### Classification Models Trained

The current script trains these three supervised models on the SMOTE-balanced training split:
- Decision Tree
- Random Forest
- XGBoost

All three models are evaluated on the original `X_test` and `y_test`.

### Segmentation Model Trained

The same script also trains:
- K-Means with `n_clusters=3`

K-Means uses `X_train_scaled.csv`, not the unscaled classification matrix.

### Files Produced by `train_models.py`

Saved models:
- `artifacts/models/decision_tree.pkl`
- `artifacts/models/random_forest.pkl`
- `artifacts/models/xgboost.pkl`
- `artifacts/models/kmeans.pkl`

Saved evaluation files:
- `artifacts/decision_tree_confusion_matrix.csv`
- `artifacts/random_forest_confusion_matrix.csv`
- `artifacts/xgboost_confusion_matrix.csv`
- `data/rf_feature_importance.csv`
- `data/xgb_feature_importance.csv`
- `data/model_comparison.csv`
- `data/train_clusters.csv`

## What `train_models.py` Does Internally

### 1. Load processed data

It loads:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`
- `X_train_scaled.csv`
- `X_test_scaled.csv`
- `target_encoder.pkl`

### 2. Rebalance the training set

It prints the original class distribution and then applies SMOTENC to create a balanced training dataset.

### 3. Train the three classifiers

Each classifier is fit on the rebalanced training data:
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `xgboost.XGBClassifier`

### 4. Evaluate on the untouched test set

For each classifier, the script calculates:
- Accuracy
- Balanced Accuracy
- Macro Precision
- Weighted Precision
- Macro F1
- MCC
- Weighted ROC-AUC
- Per-class ROC-AUC
- Classification report
- Confusion matrix

### 5. Cross-validation on the balanced training data

The script also runs stratified 5-fold cross-validation on the SMOTE-balanced training data and reports:
- Mean CV Macro F1
- Standard deviation of CV Macro F1
- Mean CV Balanced Accuracy

This is useful as a consistency check, but it is not a perfect estimate of generalization because SMOTE is applied before the folds are created.

### 6. Compare models and select the best classifier

The comparison table is sorted using:
- Primary metric: Macro F1
- Tie-breaker 1: Balanced Accuracy
- Tie-breaker 2: Weighted ROC-AUC

### 7. Train K-Means for segmentation

K-Means is then fit on `X_train_scaled`, and the silhouette score is reported.

## Metric Guide

The current training script prints several metrics. They do not all answer the same question.

### Accuracy

Definition:
- The proportion of all predictions that are correct.

What it means here:
- If accuracy is high, the model is usually correct overall.
- In an imbalanced dataset, accuracy can still look strong even if the model performs badly on minority classes.

Why it matters in this project:
- Type 2 diabetes is the majority class, so a model can get a good accuracy by leaning too heavily toward that class.

### Balanced Accuracy

Definition:
- The average recall across all classes.

What it means here:
- Every class contributes equally, regardless of how common it is.
- This makes it more fair than plain accuracy for imbalanced multiclass data.

Why it matters in this project:
- It helps reveal whether the model is ignoring rare classes such as Gestational or Type 1 diabetes.

### Precision

General meaning:
- Of the cases predicted as a class, how many were actually that class?

High precision means:
- When the model predicts a class, it is usually right.

### Macro Precision

Definition:
- Precision is computed separately for each class, then averaged equally.

What it means here:
- Every class matters the same, even if one class is very rare.

Why it matters:
- It penalizes a model that is only precise on the majority classes.

### Weighted Precision

Definition:
- Precision is computed separately for each class, then averaged using class frequency weights.

What it means here:
- Common classes contribute more to the final number.

Why it matters:
- It is useful for overall prediction quality, but it can still hide weak minority-class precision.

### Recall

General meaning:
- Of the actual members of a class, how many did the model correctly find?

Why it matters here:
- For minority classes, recall is often the first thing to collapse when the model is biased toward the majority class.

### F1 Score

Definition:
- The harmonic mean of precision and recall.

What it means:
- A model needs both decent precision and decent recall to get a good F1 score.

### Macro F1

Definition:
- F1 is computed for each class, then averaged equally across classes.

What it means here:
- It is a strong summary metric for imbalanced multiclass classification because it does not let the biggest class dominate the score.

Why it is important in this project:
- `train_models.py` uses Macro F1 as the primary model selection metric.
- This makes sense because the project is not only about predicting the majority class correctly.

### MCC

MCC stands for Matthews Correlation Coefficient.

Definition:
- A single summary score that measures how well predictions align with the true labels.

Range:
- `1`: perfect prediction
- `0`: no better than random behavior
- `-1`: total disagreement

Why it matters:
- MCC is robust when classes are imbalanced.
- It is a stronger single-number summary than raw accuracy when class distributions are uneven.

### ROC-AUC

ROC-AUC measures how well the model separates classes using predicted probabilities.

In this project the script reports:
- Weighted ROC-AUC (overall multiclass OvR)
- Per-class ROC-AUC (OvR)

OvR means One-vs-Rest:
- Each class is evaluated against all the other classes combined.

### Weighted ROC-AUC

Definition:
- A multiclass ROC-AUC averaged with class-frequency weights.

What it means here:
- Higher values mean the predicted probabilities separate classes better overall.
- Because it is weighted, common classes influence the score more.

### Per-Class ROC-AUC

Definition:
- The ROC-AUC is reported separately for each class.

Why it matters:
- It shows whether the model separates rare classes properly.
- A good overall ROC-AUC can still hide poor minority-class performance, so the per-class values are important.

### Classification Report

The classification report prints per-class:
- Precision
- Recall
- F1-score
- Support

What support means:
- The number of true examples for that class in the test set.

Why it matters:
- It is the quickest way to see whether a model is failing on specific diabetes stages.

### Confusion Matrix

Definition:
- A table showing true classes against predicted classes.

What it means here:
- The diagonal values are correct predictions.
- Off-diagonal values show which classes the model confuses with each other.

Why it matters:
- It helps identify systematic mistakes, such as predicting Pre-Diabetes when the patient is actually Type 2.

### Cross-Validation Mean and Standard Deviation

The script prints:
- Mean CV Macro F1
- Standard deviation of CV Macro F1
- Mean CV Balanced Accuracy

What they mean:
- The mean shows average performance across folds.
- The standard deviation shows stability. Lower variation suggests more consistent behavior across folds.

### Silhouette Score

This metric is for K-Means, not for the classifiers.

Definition:
- Measures how well each point fits within its own cluster compared with other clusters.

General interpretation:
- Higher is better.
- A higher silhouette score suggests clearer cluster separation.

Why it matters here:
- It is the main clustering quality check currently used in `train_models.py`.

## Current Selection Rule

The current classifier winner is selected by:
- Highest Macro F1 first
- Then Balanced Accuracy
- Then Weighted ROC-AUC

This is a reasonable rule for this project because the class distribution is imbalanced and the goal is not only to maximize overall accuracy.

## Summary of the Pipeline

1. `src/prepare_data.py` splits the raw data and creates EDA plots.
2. `src/preprocess_data.py` removes leakage, encodes the target and features, scales the data for K-Means, and saves reusable preprocessing artifacts.
3. `src/train_models.py` balances the training split with SMOTENC, trains Decision Tree, Random Forest, XGBoost, evaluates them on the untouched test set, compares them by Macro F1, and then trains K-Means for segmentation.
