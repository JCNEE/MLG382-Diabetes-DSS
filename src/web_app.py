import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
MODELS_DIR = ARTIFACTS_DIR / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

# Load models and artifacts
with open(MODELS_DIR / 'random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open(ARTIFACTS_DIR / 'target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)
with open(ARTIFACTS_DIR / 'feature_encoders.pkl', 'rb') as f:
    feature_encoders = pickle.load(f)
with open(ARTIFACTS_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(MODELS_DIR / 'kmeans.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Feature columns
X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
FEATURE_COLS = X_train.columns.tolist()

# Build dropdown options dynamically from encoders
categorical_options = {}
for col, encoder in feature_encoders.get("nominal", {}).items():
    categorical_options[col] = [{"label": c, "value": c} for c in encoder.classes_]

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Diabetes Risk Decision Support System"), className="mb-4")
    ]),

    dcc.Tabs([
        dcc.Tab(label="Risk Prediction", children=[
            html.Div([
                html.H4("Enter Patient Data"),
                *[
                    html.Div([
                        html.Label(f"{col}"),
                        dcc.Dropdown(
                            id=f"input-{col}",
                            options=categorical_options.get(col, []),
                            placeholder=f"Select {col}"
                        ) if col in categorical_options else dcc.Input(
                            id=f"input-{col}", type="number", step=0.1
                        )
                    ]) for col in FEATURE_COLS
                ]
            ]),
            html.Button("Predict Risk", id="predict-btn", n_clicks=0),
            html.Div(id="prediction-output")
        ]),

        dcc.Tab(label="Patient Segmentation", children=[
            html.H4("Cluster Assignment"),
            html.Button("Assign Cluster", id="cluster-btn", n_clicks=0),
            html.Div(id="cluster-output")
        ]),

        dcc.Tab(label="SHAP Insights", children=[
            html.H4("Model Driver Analysis"),
            html.Img(src="/assets/RandomForest_shap_summary.png", style={"width":"80%"}),
            html.Img(src="/assets/RandomForest_shap_bar.png", style={"width":"80%"})
        ]),

        dcc.Tab(label="Recommendations", children=[
            html.H4("Lifestyle Recommendations"),
            html.Div(id="recommendations-output")
        ])
    ])
])

# Callbacks
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{col}", "value") for col in FEATURE_COLS]
)
def predict_risk(n_clicks, *values):
    if n_clicks > 0:
        patient_data = pd.DataFrame([values], columns=FEATURE_COLS)

        # Apply categorical encoders
        for col, encoder in feature_encoders.get("nominal", {}).items():
            if col in patient_data.columns and patient_data[col].iloc[0] is not None:
                try:
                    patient_data[col] = encoder.transform([patient_data[col].iloc[0]])
                except ValueError:
                    patient_data[col] = -1  # unseen label → -1

        # Fill missing numeric values with training means
        train_means = pd.read_csv(DATA_DIR / 'X_train.csv').mean()
        patient_data = patient_data.fillna(train_means)

        y_pred = rf_model.predict(patient_data)
        risk_class = target_encoder.inverse_transform(y_pred)[0]
        return f"Predicted Diabetes Stage: {risk_class}"
    return ""

@app.callback(
    Output("cluster-output", "children"),
    Input("cluster-btn", "n_clicks"),
    [State(f"input-{col}", "value") for col in FEATURE_COLS]
)
def assign_cluster(n_clicks, *values):
    if n_clicks > 0:
        patient_data = pd.DataFrame([values], columns=FEATURE_COLS)

        # Apply categorical encoders
        for col, encoder in feature_encoders.get("nominal", {}).items():
            if col in patient_data.columns and patient_data[col].iloc[0] is not None:
                try:
                    patient_data[col] = encoder.transform([patient_data[col].iloc[0]])
                except ValueError:
                    patient_data[col] = -1  # unseen label → -1

        # Fill missing numeric values with training means
        train_means = pd.read_csv(DATA_DIR / 'X_train.csv').mean()
        patient_data = patient_data.fillna(train_means)

        # Scale
        patient_scaled = scaler.transform(patient_data)

        # Predict cluster
        cluster = kmeans_model.predict(patient_scaled)[0]
        return f"Assigned to Cluster: {cluster}"
    return ""

@app.callback(
    Output("recommendations-output", "children"),
    Input("cluster-output", "children")
)
def generate_recommendations(cluster_text):
    if cluster_text:
        cluster_id = int(cluster_text.split(":")[-1])
        profiles = pd.read_csv(DATA_DIR / 'cluster_profiles.csv')
        profile = profiles.loc[cluster_id]
        return html.Div([
            html.P(f"Cluster {cluster_id} Profile:"),
            html.P(f"Average Physical Activity: {profile['physical_activity_minutes_per_week']:.1f} min/week"),
            html.P(f"Average Diet Score: {profile['diet_score']:.1f}"),
            html.P("Recommendation: Increase activity and improve diet quality if below healthy thresholds.")
        ])
    return ""

if __name__ == "__main__":
    app.run(debug=True)