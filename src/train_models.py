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

# ==============================================================
# Patient Segmentation - K-Means Clustering
# ==============================================================

def train_kmeans(X_train_scaled, feature_names):
    """Train K-Means clustering with K=3"""
    print("\n" + "="*60)
    print(" K-MEANS CLUSTERING (Patient Segmentation)")
    print("="*60)
    print("   Clustering patients into 3 lifestyle/health segments")
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train_scaled)
    
    sil_score = silhouette_score(X_train_scaled, cluster_labels)
    
    print(f"\nSilhouette Score: {sil_score:.4f}")
    
    print("\n Cluster Distribution:")
    cluster_counts = np.bincount(cluster_labels)
    for i, count in enumerate(cluster_counts):
        percentage = count / len(cluster_labels) * 100
        print(f"   Cluster {i}: {count:,} patients ({percentage:.1f}%)")
    
    with open(MODELS_DIR / 'kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"\n Model saved: {MODELS_DIR / 'kmeans.pkl'}")
    
    pd.Series(cluster_labels, name='cluster').to_csv(DATA_DIR / 'train_clusters.csv', index=False)
    
    return kmeans, cluster_labels, sil_score

def compare_models(results):
    """Compare all three classification models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison = pd.DataFrame([
        {'Model': 'Decision Tree', 'Accuracy': results['Decision Tree']},
        {'Model': 'Random Forest', 'Accuracy': results['Random Forest']},
        {'Model': 'XGBoost', 'Accuracy': results['XGBoost']}
    ]).sort_values('Accuracy', ascending=False)
    
    print(comparison.to_string(index=False))
    
    best_model = comparison.iloc[0]['Model']
    best_accuracy = comparison.iloc[0]['Accuracy']
    
    print(f"\n BEST MODEL: {best_model} with {best_accuracy:.4f} accuracy")
    
    comparison.to_csv(ARTIFACTS_DIR / 'model_comparison.csv', index=False)
    
    return best_model, best_accuracy

# ==============================================================
# Main Execution
# ==============================================================

def main():
    print("="*60)
    print("DIABETES DECISION SUPPORT SYSTEM")
    print("="*60)
    print("\n Tasks:")
    print("   1. Risk Classification (Decision Tree, Random Forest, XGBoost)")
    print("   2. Patient Segmentation (K-Means with K=3)")
    
    # Load data
    data = load_data()
    
    # ==========================================================
    # PART 1: RISK CLASSIFICATION
    # ==========================================================
    print("\n" + "="*60)
    print("PART 1: RISK CLASSIFICATION")
    print("="*60)
    
    results = {}
    
    # Train Decision Tree
    dt_model, dt_acc = train_decision_tree(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        data['target_encoder']
    )
    results['Decision Tree'] = dt_acc
    
    # Train Random Forest
    rf_model, rf_acc = train_random_forest(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        data['target_encoder']
    )
    results['Random Forest'] = rf_acc
    
    # Train XGBoost
    xgb_model, xgb_acc = train_xgboost(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        data['target_encoder']
    )
    results['XGBoost'] = xgb_acc
    
    # Compare models
    best_model, best_accuracy = compare_models(results)
    
    # ==========================================================
    # PART 2: PATIENT SEGMENTATION
    # ==========================================================
    print("\n" + "="*60)
    print("PART 2: PATIENT SEGMENTATION")
    print("="*60)
    
    kmeans_model, cluster_labels, sil_score = train_kmeans(
        data['X_train_scaled'],
        data['X_train'].columns.tolist()
    )
    
    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================
    print("\n" + "="*60)
    print(" TRAINING COMPLETE!")
    print("="*60)
    
    print("\n MODELS SAVED:")
    print(f"   {MODELS_DIR}")
    print(f"   ├── decision_tree.pkl")
    print(f"   ├── random_forest.pkl")
    print(f"   ├── xgboost.pkl")
    print(f"   └── kmeans.pkl")
    
    print("\nPERFORMANCE SUMMARY:")
    print(f"    Best Risk Classifier: {best_model} ({best_accuracy:.4f})")
    print(f"    K-Means Silhouette Score: {sil_score:.4f}")

if __name__ == '__main__':
    main()