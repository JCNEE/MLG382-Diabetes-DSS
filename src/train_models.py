import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==============================================================
#Setup paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
MODELS_DIR = ARTIFACTS_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================
#Load preprocessed data
# ==============================================================

def load_data():
    """Load all preprocessed data"""
    
    
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train.csv').squeeze()
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    X_train_scaled = pd.read_csv(DATA_DIR / 'X_train_scaled.csv')
    X_test_scaled = pd.read_csv(DATA_DIR / 'X_test_scaled.csv')
    
    # Load target encoder
    with open(ARTIFACTS_DIR / 'target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test: {y_test.shape}")
    print(f"   Target classes: {list(target_encoder.classes_)}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'target_encoder': target_encoder,
    }

# ==============================================================
#Risk Classification Models
# ==============================================================

def train_decision_tree(X_train, X_test, y_train, y_test, target_encoder):
    """Train and evaluate Decision Tree"""
    print("\n" + "="*60)
    print("DECISION TREE CLASSIFIER")
    print("="*60)
    
    # Train the models
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    
    # Predict
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Results
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    
    # Save model
    with open(MODELS_DIR / 'decision_tree.pkl', 'wb') as f:
        pickle.dump(dt, f)
    print(f"\nModel saved: {MODELS_DIR / 'decision_tree.pkl'}")
    
    return dt, accuracy

def train_random_forest(X_train, X_test, y_train, y_test, target_encoder):
    """Train and evaluate Random Forest"""
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Train
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Results
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    with open(MODELS_DIR / 'random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print(f"\n Model saved: {MODELS_DIR / 'random_forest.pkl'}")
    
    # Save feature importance
    feature_importance.to_csv(ARTIFACTS_DIR / 'rf_feature_importance.csv', index=False)
    
    return rf, accuracy

def train_xgboost(X_train, X_test, y_train, y_test, target_encoder):
    """Train and evaluate XGBoost"""
    print("\n" + "="*60)
    print(" XGBOOST CLASSIFIER")
    print("="*60)
    
    # Train
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Predict
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Results
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    with open(MODELS_DIR / 'xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"\n Model saved: {MODELS_DIR / 'xgboost.pkl'}")
    
    # Save feature importance
    feature_importance.to_csv(ARTIFACTS_DIR / 'xgb_feature_importance.csv', index=False)
    
    return xgb_model, accuracy


