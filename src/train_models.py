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
