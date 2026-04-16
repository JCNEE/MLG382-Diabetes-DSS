import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from pathlib import Path

#==============================================================
# 1.Setup paths
#==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / 'data'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'


#==============================================================
# 2.Setup the columns and target variable
#==============================================================


OUTPUT = 'diabetes_stage'

LEAKAGE_COLS = ['diagnosed_diabetes', 'diabetes_risk_score']

ORDINAL_COLS = {
    'education_level': ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
    'income_level':    ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
}

NOMINAL_COLS = [
    'gender',
    'ethnicity',
    'employment_status',
    'smoking_status',]

INTERVAL_COLS = []

BINARY_COLS = [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
]

NUMERIC_COLS = [
    'Age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'glucose_fasting',
    'glucose_postprandial',
    'insulin_level',
    'hba1c',
]

FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + list(ORDINAL_COLS.keys()) + NOMINAL_COLS

#==============================================================
# 3.now we can load the split data
#==============================================================


def load_split_data():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test  = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"Loaded  train: {train.shape},  test: {test.shape}")
    return train, test

#==============================================================
# 4. drop leaking columns
#==============================================================
    
#drop leakage collumns
def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:

    return df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

#==============================================================
# 5.encode the data
#==============================================================


# encode the target data
def encode_target(train: pd.DataFrame, test: pd.DataFrame):
    le = LabelEncoder()
    y_train = le.fit_transform(train[OUTPUT])
    y_test = le.transform(test[OUTPUT])
        
    with open(ARTIFACTS_DIR / "target_encoder.pkl", "wb") as f:
        pickle.dump(le, f)    

    return y_train, y_test, le
    
#encode categorical data
def encode_categorical(train: pd.DataFrame, test: pd.DataFrame):

    X_train = train[FEATURE_COLS].copy()
    X_test  = test[FEATURE_COLS].copy()

    encoders = {}

    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_COLS[col] for col in ORDINAL_COLS],
        handle_unknown='use_encoded_value',
        unknown_value=-1
        )
    ordinal_col_list = list(ORDINAL_COLS.keys())
    X_train[ordinal_col_list] = ordinal_encoder.fit_transform(X_train[ordinal_col_list])
    X_test[ordinal_col_list]  = ordinal_encoder.transform(X_test[ordinal_col_list])
    encoders['ordinal'] = ordinal_encoder

    nominal_encoders = {}
    for col in NOMINAL_COLS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col]  = le.transform(X_test[col])
        nominal_encoders[col] = le
    encoders['nominal'] = nominal_encoders

    with open(ARTIFACTS_DIR / 'feature_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    print(f"Feature matrix shape – train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, encoders    

#==============================================================
# 6. scaling the data for K-means and SVM
#==============================================================


def Scaling(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

    with open(ARTIFACTS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Scaled feature matrix ready for K-Means.")
    return X_train_scaled, X_test_scaled, scaler

#==============================================================
# 7. Saving the data that has been processed for modeling
#==============================================================


def Save_data(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):

    X_train.to_csv(DATA_DIR / 'X_train.csv', index=False)
    X_test.to_csv(DATA_DIR / 'X_test.csv', index=False)


    pd.Series(y_train, name=OUTPUT).to_csv(DATA_DIR / 'y_train.csv', index=False)
    pd.Series(y_test,  name=OUTPUT).to_csv(DATA_DIR / 'y_test.csv',  index=False)

    X_train_scaled.to_csv(DATA_DIR / 'X_train_scaled.csv', index=False)
    X_test_scaled.to_csv(DATA_DIR  / 'X_test_scaled.csv',  index=False)


    print("Processed data saved to artifacts directory.")

#==============================================================
# 7. Main pipeline
#==============================================================
#this runs the functions that was created, encodes the data for the models, and saves the data

def preprocess_data():
    #loads the data that was split in the "prepare data" file
    train, test = load_split_data()

    #removes the column leakage (which negatively affects the models' performance)
    train = drop_leakage(train)
    test = drop_leakage(test)

    # here the target (aka output) is encoded for the models
    y_train, y_test, target_encoder = encode_target(train, test)

    # here the categorical features (aka inputs) are encoded for the models
    X_train, X_test, feature_encoders = encode_categorical(train, test)

    # here the numeric features are scaled and encoded for K-means and SVM
    X_train_scaled, X_test_scaled, scaler = Scaling(X_train, X_test)

    # everything is saved
    Save_data(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

    return {
        # Unscaled (use for training the 4 models DT / RF / XGBoost / SHAP)
        'X_train':        X_train,
        'X_test':         X_test,
        'y_train':        y_train,
        'y_test':         y_test,
        # Scaled (This will be used for K-Means and SVM)
        'X_train_scaled': X_train_scaled,
        'X_test_scaled':  X_test_scaled,
        # Encoders (these will be used in the dashboard/interface))
        'target_encoder':   target_encoder,
        'feature_encoders': feature_encoders,
        'scaler':           scaler,
    }

#This will only run of the file is executed directly, not if it's imported as a module.

if __name__ == '__main__':
    artefacts = preprocess_data()
    print("\n=== Preprocessing complete ===")
    print(f"X_train shape : {artefacts['X_train'].shape}")
    print(f"y_train shape : {artefacts['y_train'].shape}")
    print(f"X_train_scaled: {artefacts['X_train_scaled'].shape}")
