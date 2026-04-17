import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from imblearn.over_sampling import SMOTENC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,      
    precision_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    confusion_matrix,
    matthews_corrcoef,            
)
from sklearn.model_selection import StratifiedKFold, cross_validate   
import xgboost as xgb
import warnings

try:
    from preprocess_data import BINARY_COLS, ORDINAL_COLS, NOMINAL_COLS
except ImportError:
    from src.preprocess_data import BINARY_COLS, ORDINAL_COLS, NOMINAL_COLS

warnings.filterwarnings('ignore')


# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
MODELS_DIR = ARTIFACTS_DIR / 'models'
ASSETS_DIR = PROJECT_ROOT / 'assets'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PRIMARY_CLASSIFICATION_METRIC = 'Macro F1'
CLUSTER_PROFILE_COLUMNS = [
    'physical_activity_minutes_per_week',
    'diet_score',
    'bmi',
    'glucose_fasting',
    'hba1c',
]
CLUSTER_SEGMENT_LABELS = {
    'healthy': 'Healthy Patient Cluster',
    'elevated_glucose': 'Elevated Glucose Patient Cluster',
    'unhealthy': 'Unhealthy Patient Cluster',
}
CLUSTER_SEGMENT_COLORS = {
    'healthy': '#2f8f68',
    'elevated_glucose': '#d7a255',
    'unhealthy': '#c9774c',
    'unknown': '#91a196',
}
CLUSTER_IMAGE_MAX_POINTS = 12000
CLUSTER_SIZE_IMAGE = 'patient_segmentation_cluster_sizes.png'
CLUSTER_MAP_IMAGE = 'patient_segmentation_cluster_map.png'

CORE_MODEL_BUILDERS = {
    'Decision Tree': lambda: DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=20,
        min_samples_split=40,
        random_state=42,
    ),
    'Random Forest': lambda: RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    ),
    'XGBoost': lambda: xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1,
    ),
}

EXTENDED_NOTEBOOK_MODEL_BUILDERS = {
    'Extra Trees': lambda: ExtraTreesClassifier(
        n_estimators=250,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    ),
}


def get_model_builders(include_extended=False):
    """Return the shared model registry used by the training script and notebook."""
    model_builders = dict(CORE_MODEL_BUILDERS)
    if include_extended:
        model_builders.update(EXTENDED_NOTEBOOK_MODEL_BUILDERS)
    return model_builders

#==============================================================
# The following function loads the preprocessed training and 
# testing data, along with the target encoder, and prints 
# their shapes and class distribution.
#==============================================================
def load_data():
    """Load all preprocessed data."""
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train.csv').squeeze()
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    X_train_scaled = pd.read_csv(DATA_DIR / 'X_train_scaled.csv')
    X_test_scaled = pd.read_csv(DATA_DIR / 'X_test_scaled.csv')

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



def print_class_distribution(y_values, target_encoder, heading):
    """Print class counts alongside the encoded class names."""
    print(f"\n{heading}")
    counts = pd.Series(y_values).value_counts().sort_index()
    total = len(y_values)
    for class_index, class_name in enumerate(target_encoder.classes_):
        count = int(counts.get(class_index, 0))
        percentage = (count / total * 100) if total else 0
        print(f"   {class_name}: {count:,} ({percentage:.1f}%)")

#====================================
# encoding the features for modeling
#====================================
def balance_training_data(X_train, y_train, target_encoder):

    print("\n" + "=" * 60)
    print("TRAINING DATA REBALANCING")
    print("=" * 60)
    print("   Applying SMOTENC to the training split only")

    print_class_distribution(y_train, target_encoder, "Original class distribution:")

    class_counts = pd.Series(y_train).value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError(
            "SMOTE-based balancing requires at least two samples in every class."
        )

    categorical_cols = BINARY_COLS + list(ORDINAL_COLS.keys()) + NOMINAL_COLS
    categorical_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_cols
        if col in X_train.columns
    ]

    smote = SMOTENC(
        categorical_features=categorical_indices,
        random_state=42,
        k_neighbors=min(5, min_class_count - 1),
    )
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    if not isinstance(X_balanced, pd.DataFrame):
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
    X_balanced = X_balanced.astype(float)

    if not isinstance(y_balanced, pd.Series):
        y_balanced = pd.Series(
            y_balanced, name=getattr(y_train, 'name', 'diabetes_stage')
        )

    print_class_distribution(y_balanced, target_encoder, "Resampled class distribution:")
    return X_balanced, y_balanced

#==============================================================
# The following line defines the list of feature columns by 
# combining numeric, binary, ordinal, and nominal columns.
#==============================================================
def evaluate_classifier_outputs(
    model_name,
    model,
    X_test,
    y_test,
    target_encoder,
    X_train_balanced=None,
    y_train_balanced=None,
    *,
    save_confusion_csv=True,
    verbose=True,
):
    """Return notebook-friendly model outputs while preserving the script metrics."""
    class_labels = np.arange(len(target_encoder.classes_))
    y_pred = model.predict(X_test)
    proba_frame = pd.DataFrame(
        model.predict_proba(X_test), columns=model.classes_
    )
    y_proba = proba_frame.reindex(columns=class_labels, fill_value=0.0).to_numpy()


    # ── Core metrics ──────────────────────────────────────────────────────────
    accuracy           = accuracy_score(y_test, y_pred)
    # FIX: balanced accuracy treats each class equally regardless of size
    bal_accuracy       = balanced_accuracy_score(y_test, y_pred)
    weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_precision    = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1           = f1_score(y_test, y_pred, average='macro', zero_division=0)
    # FIX: MCC — reliable single-value summary for imbalanced multiclass
    mcc                = matthews_corrcoef(y_test, y_pred)
    weighted_roc_auc   = roc_auc_score(
        y_test, y_proba, labels=class_labels, multi_class='ovr', average='weighted'
    )
    # FIX: per-class ROC-AUC (OvR) reveals minority-class separation quality
    per_class_roc_auc = roc_auc_score(
        y_test, y_proba, labels=class_labels, multi_class='ovr', average=None
    )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    confusion = confusion_matrix(y_test, y_pred, labels=class_labels)
    confusion_df = pd.DataFrame(
        confusion,
        index=target_encoder.classes_,
        columns=target_encoder.classes_,
    )
    confusion_path = None
    if save_confusion_csv:
        confusion_path = (
            ARTIFACTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
        )
        confusion_df.to_csv(confusion_path)

    report_text = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=target_encoder.classes_,
        zero_division=0,
    )
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            target_names=target_encoder.classes_,
            zero_division=0,
            output_dict=True,
        )
    ).transpose()


    # ── Print results ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\nMacro F1:              {macro_f1:.4f}")
        print(f"Accuracy:              {accuracy:.4f}")
        # FIX: highlight when balanced accuracy diverges from plain accuracy — that gap
        # directly measures how much the model is coasting on majority-class prevalence.
        print(f"Balanced Accuracy:     {bal_accuracy:.4f}  ← fairer metric for imbalanced data")
        print(f"MCC:                   {mcc:.4f}  ← -1 worst | 0 random | 1 perfect")
        print(f"Macro Precision:       {macro_precision:.4f}")
        print(f"Weighted Precision:    {weighted_precision:.4f}")
        print(f"Weighted ROC-AUC(OvR): {weighted_roc_auc:.4f}")

        # FIX: per-class ROC-AUC — key for spotting if Type 1 / Gestational are ignored
        print("\nPer-Class ROC-AUC (OvR):")
        for cls_name, auc_val in zip(target_encoder.classes_, per_class_roc_auc):
            print(f"   {cls_name}: {auc_val:.4f}")

        print("\nClassification Report:")
        print(report_text)
        print("\nConfusion Matrix:")
        print(confusion_df.to_string())
        if confusion_path is not None:
            print(f"\nConfusion matrix saved: {confusion_path}")

    # ── Optional CV (FIX) ─────────────────────────────────────────────────────
    # Cross-validate on SMOTE-balanced training data.
    # WARNING: CV on SMOTE data measures consistency, NOT generalisation to the
    # real-world imbalanced distribution.  Treat it as an overfitting indicator.
    cv_results = None
    if X_train_balanced is not None and y_train_balanced is not None:
        if verbose:
            print("\nStratified 5-Fold CV (on SMOTE-balanced training data):")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv = cross_validate(
            model, X_train_balanced, y_train_balanced,
            cv=skf,
            scoring={
                'macro_f1': 'f1_macro',
                'balanced_accuracy': 'balanced_accuracy',
            },
            n_jobs=-1,
        )
        cv_macro_f1  = cv['test_macro_f1']
        cv_bal_acc   = cv['test_balanced_accuracy']
        if verbose:
            print(f"   Macro F1:          {cv_macro_f1.mean():.4f} ± {cv_macro_f1.std():.4f}")
            print(f"   Balanced Accuracy: {cv_bal_acc.mean():.4f} ± {cv_bal_acc.std():.4f}")
        cv_results = {
            'CV Macro F1 Mean': cv_macro_f1.mean(),
            'CV Macro F1 Std':  cv_macro_f1.std(),
        }

    metrics = {
        'Model':              model_name,
        'Macro F1':           macro_f1,
        'Accuracy':           accuracy,
        'Balanced Accuracy':  bal_accuracy,     # FIX: added
        'MCC':                mcc,              # FIX: added
        'Macro Precision':    macro_precision,
        'Weighted Precision': weighted_precision,
        'Weighted ROC-AUC':   weighted_roc_auc,
    }
    if cv_results:
        metrics.update(cv_results)

    return {
        'metrics': metrics,
        'confusion': confusion,
        'confusion_df': confusion_df,
        'report': report_df,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_path': confusion_path,
    }


def evaluate_classifier(model_name, model, X_test, y_test, target_encoder,
                         X_train_balanced=None, y_train_balanced=None):
    evaluation = evaluate_classifier_outputs(
        model_name,
        model,
        X_test,
        y_test,
        target_encoder,
        X_train_balanced,
        y_train_balanced,
        save_confusion_csv=True,
        verbose=True,
    )
    return evaluation['metrics']


def run_modeling_experiments(
    X_fit,
    y_fit,
    X_test,
    y_test,
    target_encoder,
    *,
    include_extended=False,
    model_builders=None,
):
    """Run the shared notebook model comparison without rewriting evaluation logic."""
    builders = model_builders or get_model_builders(include_extended=include_extended)
    evaluations = {}
    comparison_rows = []

    for model_name, build_model in builders.items():
        model = build_model()
        model.fit(X_fit, y_fit)
        evaluation = evaluate_classifier_outputs(
            model_name,
            model,
            X_test,
            y_test,
            target_encoder,
            save_confusion_csv=False,
            verbose=False,
        )
        evaluation['model'] = model
        evaluations[model_name] = evaluation
        comparison_rows.append(evaluation['metrics'])

    comparison = pd.DataFrame(comparison_rows).sort_values(
        [PRIMARY_CLASSIFICATION_METRIC, 'Balanced Accuracy', 'Weighted ROC-AUC'],
        ascending=False,
    ).reset_index(drop=True)
    return evaluations, comparison

#============================
# Risk Classification Models
#============================
def train_decision_tree(X_train, X_test, y_train, y_test, target_encoder):
    """Train and evaluate Decision Tree."""
    print("\n" + "=" * 60)
    print("DECISION TREE CLASSIFIER")
    print("=" * 60)

    # NOTE: class_weight is NOT set here because SMOTENC already balanced the
    # training data.  Adding class_weight='balanced' on top of SMOTE would
    # double-penalise the majority classes and hurt overall precision.
    dt = CORE_MODEL_BUILDERS['Decision Tree']()
    dt.fit(X_train, y_train)

    metrics = evaluate_classifier(
        'Decision Tree', dt, X_test, y_test, target_encoder,
        X_train, y_train,
    )

    with open(MODELS_DIR / 'decision_tree.pkl', 'wb') as f:
        pickle.dump(dt, f)
    print(f"\nModel saved: {MODELS_DIR / 'decision_tree.pkl'}")

    return dt, metrics


def train_random_forest(X_train, X_test, y_train, y_test, target_encoder):
    """Train and evaluate Random Forest."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    rf = CORE_MODEL_BUILDERS['Random Forest']()
    rf.fit(X_train, y_train)

    metrics = evaluate_classifier(
        'Random Forest', rf, X_test, y_test, target_encoder,
        X_train, y_train,
    )

    # FIX: use enumerate so rank numbers are always 1–10 regardless of DataFrame index
    feature_importance = pd.DataFrame({
        'feature':    X_train.columns,
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    for rank, (_, row) in enumerate(feature_importance.head(10).iterrows(), start=1):
        print(f"   {rank}. {row['feature']}: {row['importance']:.4f}")

    with open(MODELS_DIR / 'random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print(f"\nModel saved: {MODELS_DIR / 'random_forest.pkl'}")

    feature_importance.to_csv(DATA_DIR / 'rf_feature_importance.csv', index=False)
    return rf, metrics


def train_xgboost(X_train, X_test, y_train, y_test, target_encoder):
    """
    Train and evaluate XGBoost.
    SMOTENC has already balanced the class distribution, so no sample_weight
    or class_weight is applied — that would double-balance and distort precision.
    """
    print("\n" + "=" * 60)
    print("XGBOOST CLASSIFIER")
    print("=" * 60)

    xgb_model = CORE_MODEL_BUILDERS['XGBoost']()


    xgb_model.fit(X_train, y_train)

    metrics = evaluate_classifier(
        'XGBoost', xgb_model, X_test, y_test, target_encoder,
        X_train, y_train,
    )


    feature_importance = pd.DataFrame({
        'feature':    X_train.columns,
        'importance': xgb_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    for rank, (_, row) in enumerate(feature_importance.head(10).iterrows(), start=1):
        print(f"   {rank}. {row['feature']}: {row['importance']:.4f}")

    with open(MODELS_DIR / 'xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"\nModel saved: {MODELS_DIR / 'xgboost.pkl'}")

    feature_importance.to_csv(DATA_DIR / 'xgb_feature_importance.csv', index=False)
    return xgb_model, metrics


#======================
# Patient Segmentation
#======================
def train_kmeans(X_train_scaled, feature_names):
    """Train K-Means clustering with K=3."""
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING (Patient Segmentation)")
    print("=" * 60)
    print("   Clustering patients into 3 lifestyle/health segments")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train_scaled)

    sil_score = silhouette_score(X_train_scaled, cluster_labels)
    print(f"\nSilhouette Score: {sil_score:.4f}")

    print("\nCluster Distribution:")
    cluster_counts = np.bincount(cluster_labels)
    for i, count in enumerate(cluster_counts):
        percentage = count / len(cluster_labels) * 100
        print(f"   Cluster {i}: {count:,} patients ({percentage:.1f}%)")

    with open(MODELS_DIR / 'kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"\nModel saved: {MODELS_DIR / 'kmeans.pkl'}")

    pd.Series(cluster_labels, name='cluster').to_csv(
        DATA_DIR / 'train_clusters.csv', index=False
    )
    return kmeans, cluster_labels, sil_score

#==============================================================
# The following function encodes the features for modeling using 
# OrdinalEncoder for ordinal features and LabelEncoder for nominal 
# features, and saves the encoders for future use.
#==============================================================
def save_cluster_profiles(X_train, cluster_labels):
    """Save cluster-level numeric summaries for dashboard recommendations."""
    available_columns = [column for column in CLUSTER_PROFILE_COLUMNS if column in X_train.columns]
    if not available_columns:
        raise ValueError('No cluster profile columns were found in X_train.')

    cluster_frame = X_train[available_columns].copy()
    cluster_frame['cluster'] = cluster_labels

    cluster_profiles = cluster_frame.groupby('cluster')[available_columns].mean().round(2)
    cluster_profiles['cluster_size'] = cluster_frame.groupby('cluster').size()

    cluster_profiles_path = DATA_DIR / 'cluster_profiles.csv'
    cluster_profiles.reset_index().to_csv(cluster_profiles_path, index=False)
    print(f"Cluster profiles saved: {cluster_profiles_path}")
    return cluster_profiles


def build_cluster_segment_keys(cluster_profiles):
    """Assign stable descriptive names to the saved K-means clusters."""
    if cluster_profiles is None or cluster_profiles.empty:
        return {}

    if 'cluster' in cluster_profiles.columns:
        profile_frame = cluster_profiles.set_index('cluster')
    else:
        profile_frame = cluster_profiles.copy()

    if any(column not in profile_frame.columns for column in CLUSTER_PROFILE_COLUMNS):
        return {}

    profiles = profile_frame[CLUSTER_PROFILE_COLUMNS].apply(pd.to_numeric, errors='coerce')
    segment_keys = {}

    overall_health_rank = (
        profiles['physical_activity_minutes_per_week'].rank(method='dense', ascending=False)
        + profiles['diet_score'].rank(method='dense', ascending=False)
        + profiles['bmi'].rank(method='dense', ascending=True)
        + profiles['glucose_fasting'].rank(method='dense', ascending=True)
        + profiles['hba1c'].rank(method='dense', ascending=True)
    )
    healthiest_cluster = int(overall_health_rank.idxmin())
    segment_keys[healthiest_cluster] = 'healthy'

    remaining_clusters = [cluster_id for cluster_id in profiles.index if int(cluster_id) not in segment_keys]
    if remaining_clusters:
        remaining_profiles = profiles.loc[remaining_clusters]
        glucose_risk_rank = (
            remaining_profiles['glucose_fasting'].rank(method='dense', ascending=False)
            + remaining_profiles['hba1c'].rank(method='dense', ascending=False)
        )
        highest_glucose_cluster = int(glucose_risk_rank.idxmin())
        segment_keys[highest_glucose_cluster] = 'elevated_glucose'

    for cluster_id in profiles.index:
        cluster_key = int(cluster_id)
        if cluster_key not in segment_keys:
            segment_keys[cluster_key] = 'unhealthy'

    return segment_keys


def sample_cluster_projection_points(projection_frame):
    if len(projection_frame) <= CLUSTER_IMAGE_MAX_POINTS:
        return projection_frame.copy(), False

    sample_ratio = CLUSTER_IMAGE_MAX_POINTS / len(projection_frame)
    sampled_frames = []
    for _, cluster_frame in projection_frame.groupby('cluster', sort=False):
        sample_size = min(
            len(cluster_frame),
            max(1, int(round(len(cluster_frame) * sample_ratio))),
        )
        sampled_frames.append(cluster_frame.sample(n=sample_size, random_state=42))

    sampled_frame = pd.concat(sampled_frames, ignore_index=True)
    if len(sampled_frame) > CLUSTER_IMAGE_MAX_POINTS:
        sampled_frame = sampled_frame.sample(n=CLUSTER_IMAGE_MAX_POINTS, random_state=42)

    return sampled_frame.reset_index(drop=True), True


def get_cluster_visualization_paths():
    """Return the saved dashboard image paths for notebook display."""
    return ASSETS_DIR / CLUSTER_SIZE_IMAGE, ASSETS_DIR / CLUSTER_MAP_IMAGE


def save_cluster_visualizations(X_train_scaled, cluster_labels, cluster_profiles, kmeans_model):
    """Save static cluster-size and cluster-map images for the dashboard."""
    profile_frame = cluster_profiles.reset_index() if 'cluster' not in cluster_profiles.columns else cluster_profiles.copy()
    profile_frame['cluster'] = pd.to_numeric(profile_frame['cluster'], errors='coerce').astype(int)
    segment_keys = build_cluster_segment_keys(profile_frame)

    size_frame = profile_frame[['cluster', 'cluster_size']].copy().sort_values('cluster').reset_index(drop=True)
    size_frame['cluster_name'] = size_frame['cluster'].map(
        lambda cluster_id: CLUSTER_SEGMENT_LABELS.get(segment_keys.get(int(cluster_id)), f'Cluster {int(cluster_id)}')
    )
    size_frame['color'] = size_frame['cluster'].map(
        lambda cluster_id: CLUSTER_SEGMENT_COLORS.get(segment_keys.get(int(cluster_id), 'unknown'), CLUSTER_SEGMENT_COLORS['unknown'])
    )
    size_frame['cluster_share_pct'] = size_frame['cluster_size'] / size_frame['cluster_size'].sum() * 100

    size_path = ASSETS_DIR / CLUSTER_SIZE_IMAGE
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    bars = ax.bar(size_frame['cluster_name'], size_frame['cluster_size'], color=size_frame['color'])
    ax.set_title('Training Patients Per Segment')
    ax.set_ylabel('Training patients')
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', rotation=14)
    ax.set_ylim(0, size_frame['cluster_size'].max() * 1.14)

    for bar, row in zip(bars, size_frame.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(row.cluster_size):,}\n({row.cluster_share_pct:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(size_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    projection_model = PCA(n_components=2, random_state=42)
    components = projection_model.fit_transform(X_train_scaled)
    projection_frame = pd.DataFrame(
        {
            'pc1': components[:, 0],
            'pc2': components[:, 1],
            'cluster': pd.Series(cluster_labels).astype(int).to_numpy(),
        }
    )
    projection_frame['cluster_name'] = projection_frame['cluster'].map(
        lambda cluster_id: CLUSTER_SEGMENT_LABELS.get(segment_keys.get(int(cluster_id)), f'Cluster {int(cluster_id)}')
    )
    sampled_projection, is_sampled = sample_cluster_projection_points(projection_frame)

    centroid_points = projection_model.transform(
        pd.DataFrame(kmeans_model.cluster_centers_, columns=X_train_scaled.columns)
    )
    centroid_frame = pd.DataFrame(
        {
            'pc1': centroid_points[:, 0],
            'pc2': centroid_points[:, 1],
            'cluster': np.arange(len(centroid_points), dtype=int),
        }
    )
    centroid_frame['cluster_name'] = centroid_frame['cluster'].map(
        lambda cluster_id: CLUSTER_SEGMENT_LABELS.get(segment_keys.get(int(cluster_id)), f'Cluster {int(cluster_id)}')
    )
    centroid_frame['color'] = centroid_frame['cluster'].map(
        lambda cluster_id: CLUSTER_SEGMENT_COLORS.get(segment_keys.get(int(cluster_id), 'unknown'), CLUSTER_SEGMENT_COLORS['unknown'])
    )

    map_path = ASSETS_DIR / CLUSTER_MAP_IMAGE
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    for row in size_frame.itertuples(index=False):
        cluster_points = sampled_projection[sampled_projection['cluster'] == row.cluster]
        ax.scatter(
            cluster_points['pc1'],
            cluster_points['pc2'],
            s=10,
            alpha=0.28,
            color=row.color,
            label=row.cluster_name,
        )

    ax.scatter(
        centroid_frame['pc1'],
        centroid_frame['pc2'],
        marker='D',
        s=120,
        color=centroid_frame['color'],
        edgecolors='white',
        linewidth=1.5,
        zorder=3,
    )
    for row in centroid_frame.itertuples(index=False):
        ax.annotate(
            row.cluster_name,
            (row.pc1, row.pc2),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center',
            fontsize=9,
        )

    variance_ratio = projection_model.explained_variance_ratio_
    ax.set_title('2D PCA View of Patient Segments')
    ax.set_xlabel(f'Principal Component 1 ({variance_ratio[0] * 100:.1f}% variance)')
    ax.set_ylabel(f'Principal Component 2 ({variance_ratio[1] * 100:.1f}% variance)')
    ax.grid(alpha=0.18)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            linestyle='',
            label=row.cluster_name,
            markerfacecolor=row.color,
            markeredgecolor='white',
            markeredgewidth=0.8,
            markersize=7,
            alpha=1.0,
        )
        for row in size_frame.itertuples(index=False)
    ]
    ax.legend(
        handles=legend_handles,
        loc='best',
        frameon=True,
        facecolor='white',
        edgecolor='#d9e1db',
        framealpha=0.95,
    )

    if is_sampled:
        ax.text(
            1.0,
            -0.12,
            f'Showing {len(sampled_projection):,} sampled points for readability.',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(map_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Cluster size image saved: {size_path}")
    print(f"Cluster map image saved: {map_path}")
    return size_path, map_path

#==============================================================
# The following function compares the performance of the trained 
# classification models and saves a summary table of their metrics.
#==============================================================
def compare_models(results):
    """
    Compare all three classification models.

    FIX: comparison DataFrame now includes Balanced Accuracy and MCC so the
    summary table reflects imbalance-aware performance, not just raw accuracy.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    comparison = pd.DataFrame([
        results['Decision Tree'],
        results['Random Forest'],
        results['XGBoost'],
    ]).sort_values(
        [PRIMARY_CLASSIFICATION_METRIC, 'Balanced Accuracy', 'Weighted ROC-AUC'],
        ascending=False,
    )

    display_cols = [
        'Model', 'Macro F1', 'Balanced Accuracy', 'MCC',
        'Accuracy', 'Weighted ROC-AUC',
    ]
    # Only show columns that actually exist (CV columns are optional)
    display_cols = [c for c in display_cols if c in comparison.columns]
    print(comparison[display_cols].to_string(index=False))

    best_model        = comparison.iloc[0]['Model']
    best_metric_value = comparison.iloc[0][PRIMARY_CLASSIFICATION_METRIC]
    best_accuracy     = comparison.iloc[0]['Accuracy']
    best_bal_accuracy = comparison.iloc[0]['Balanced Accuracy']

    print(
        f"\nBEST MODEL BY {PRIMARY_CLASSIFICATION_METRIC.upper()}: "
        f"{best_model} ({best_metric_value:.4f})"
    )
    print(f"Accuracy for selected model:          {best_accuracy:.4f}")
    # FIX: report balanced accuracy alongside plain accuracy in the summary
    print(f"Balanced Accuracy for selected model: {best_bal_accuracy:.4f}")

    comparison.to_csv(DATA_DIR / 'model_comparison.csv', index=False)
    return best_model, best_metric_value, best_accuracy, best_bal_accuracy


#===================
# Main pipeline
#==================
def main():
    print("=" * 60)
    print("DIABETES DECISION SUPPORT SYSTEM")
    print("=" * 60)
    print("\nTasks:")
    print("   1. Risk Classification (Decision Tree, Random Forest, XGBoost)")
    print("   2. Patient Segmentation (K-Means with K=3)")

    data = load_data()

    # Keep a copy of the original imbalanced y_train for XGBoost sample_weight
    y_train_original = data['y_train'].copy()   # FIX: needed by train_xgboost

    X_train_balanced, y_train_balanced = balance_training_data(
        data['X_train'], data['y_train'], data['target_encoder']
    )

    # ── PART 1: RISK CLASSIFICATION ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 1: RISK CLASSIFICATION")
    print("=" * 60)

    results = {}

    dt_model, dt_metrics = train_decision_tree(
        X_train_balanced, data['X_test'],
        y_train_balanced, data['y_test'],
        data['target_encoder'],
    )
    results['Decision Tree'] = dt_metrics

    rf_model, rf_metrics = train_random_forest(
        X_train_balanced, data['X_test'],
        y_train_balanced, data['y_test'],
        data['target_encoder'],
    )
    results['Random Forest'] = rf_metrics

    xgb_model, xgb_metrics = train_xgboost(
        X_train_balanced, data['X_test'],
        y_train_balanced, data['y_test'],
        data['target_encoder'],
    )
    results['XGBoost'] = xgb_metrics

    best_model, best_macro_f1, best_accuracy, best_bal_accuracy = compare_models(results)

    # ── PART 2: PATIENT SEGMENTATION ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2: PATIENT SEGMENTATION")
    print("=" * 60)

    kmeans_model, cluster_labels, sil_score = train_kmeans(
        data['X_train_scaled'],
        data['X_train'].columns.tolist(),
    )
    cluster_profiles = save_cluster_profiles(data['X_train'], cluster_labels)
    save_cluster_visualizations(
        data['X_train_scaled'],
        cluster_labels,
        cluster_profiles,
        kmeans_model,
    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    print("\nMODELS SAVED:")
    print(f"   {MODELS_DIR}")
    for name in ('decision_tree.pkl', 'random_forest.pkl', 'xgboost.pkl', 'kmeans.pkl'):
        print(f"   {name}")

    print("\nPERFORMANCE SUMMARY:")
    print(f"   Best Risk Classifier by Macro F1:  {best_model} ({best_macro_f1:.4f})")
    print(f"   Accuracy (selected model):          {best_accuracy:.4f}")
    # FIX: surface balanced accuracy in the final summary
    print(f"   Balanced Accuracy (selected model): {best_bal_accuracy:.4f}")
    print(f"   K-Means Silhouette Score:           {sil_score:.4f}")


if __name__ == '__main__':
    main()
