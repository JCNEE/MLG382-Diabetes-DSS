import os
from pathlib import Path
import pickle

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

try:
    from preprocess_data import BINARY_COLS, FEATURE_COLS, NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS
except ImportError:
    from src.preprocess_data import BINARY_COLS, FEATURE_COLS, NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

PREDICTION_MODEL_FILE = MODELS_DIR / "xgboost.pkl"
MODEL_LABEL = "XGBoost"
MODEL_ASSET_PREFIX = "XGBoost"

FEATURE_LABELS = {
    "Age": "Age",
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "hdl_cholesterol": "HDL Cholesterol",
    "ldl_cholesterol": "LDL Cholesterol",
}

PRIMARY_FEATURE_COUNT = 10
PRIMARY_FEATURE_FALLBACK = [
    "hba1c",
    "glucose_fasting",
    "Age",
    "gender",
    "systolic_bp",
    "glucose_postprandial",
    "income_level",
    "hypertension_history",
    "family_history_diabetes",
    "hdl_cholesterol",
]


def load_pickle(path: Path):
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def load_prediction_model():
    if not PREDICTION_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Missing XGBoost model. Run src/train_models.py first."
        )
    return load_pickle(PREDICTION_MODEL_FILE)


PREDICTION_MODEL = load_prediction_model()

target_encoder = load_pickle(ARTIFACTS_DIR / "target_encoder.pkl")
feature_encoders = load_pickle(ARTIFACTS_DIR / "feature_encoders.pkl")
scaler = load_pickle(ARTIFACTS_DIR / "scaler.pkl")
kmeans_model = load_pickle(MODELS_DIR / "kmeans.pkl")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
train_feature_cols = X_train.columns.tolist()
train_numeric_means = X_train[NUMERIC_COLS].mean().to_dict()
train_fill_values = {}
for column in train_feature_cols:
    if column in NUMERIC_COLS:
        train_fill_values[column] = float(train_numeric_means[column])
    else:
        train_fill_values[column] = float(X_train[column].mode().iloc[0])

shap_background = X_train.sample(min(500, len(X_train)), random_state=42).copy()
shap_explainer = None
cluster_baselines = X_train[
    [
        "physical_activity_minutes_per_week",
        "diet_score",
        "bmi",
        "glucose_fasting",
        "hba1c",
    ]
].median().to_dict()


def load_primary_features():
    shap_path = DATA_DIR / "xgboost_shap_feature_importance.csv"
    if shap_path.exists():
        shap_frame = pd.read_csv(shap_path)
        ranked = [feature for feature in shap_frame["feature"].tolist() if feature in FEATURE_COLS]
        ranked = list(dict.fromkeys(ranked))
        if len(ranked) >= PRIMARY_FEATURE_COUNT:
            return ranked[:PRIMARY_FEATURE_COUNT]
    return [feature for feature in PRIMARY_FEATURE_FALLBACK if feature in FEATURE_COLS][:PRIMARY_FEATURE_COUNT]


PRIMARY_FEATURES = load_primary_features()
SECONDARY_FEATURES = [feature for feature in FEATURE_COLS if feature not in PRIMARY_FEATURES]


def pretty_feature_name(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    return feature_name.replace("_", " ").title()


def format_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Imputed"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)


def format_feature_list(columns: list[str], max_items: int = 4) -> str:
    if not columns:
        return "None"
    labels = [pretty_feature_name(column) for column in columns[:max_items]]
    if len(columns) > max_items:
        labels.append(f"+{len(columns) - max_items} more")
    return ", ".join(labels)


def column_requires_decimal(column: str) -> bool:
    if column not in NUMERIC_COLS:
        return False
    values = X_train[column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return True
    return not np.all(np.isclose(values, np.round(values)))


def get_numeric_step(column: str):
    return 0.1 if column_requires_decimal(column) else 1


def get_fallback_text(column: str) -> str:
    if column in NUMERIC_COLS:
        return f"Average if blank: {train_numeric_means[column]:.2f}"
    if column in BINARY_COLS:
        label = "Yes" if int(round(train_fill_values[column])) == 1 else "No"
        return f"Default if blank: {label}"
    if column in ORDINAL_COLS:
        categories = list(ORDINAL_COLS[column])
        index = int(round(train_fill_values[column]))
        if 0 <= index < len(categories):
            return f"Default if blank: {categories[index]}"
        return "Default if blank: most common value"
    if column in NOMINAL_COLS:
        encoder = feature_encoders["nominal"][column]
        index = int(round(train_fill_values[column]))
        classes = list(encoder.classes_)
        if 0 <= index < len(classes):
            return f"Default if blank: {classes[index]}"
        return "Default if blank: most common value"
    return "Default if blank: training baseline"


def build_empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
    )
    return figure


def get_input_options(column: str):
    if column in BINARY_COLS:
        return [
            {"label": "No", "value": 0},
            {"label": "Yes", "value": 1},
        ]
    if column in ORDINAL_COLS:
        return [{"label": category, "value": category} for category in ORDINAL_COLS[column]]
    if column in NOMINAL_COLS:
        encoder = feature_encoders["nominal"][column]
        return [{"label": category, "value": category} for category in encoder.classes_]
    return None


def build_input_field(column: str):
    options = get_input_options(column)
    label = pretty_feature_name(column)
    helper_text = get_fallback_text(column)
    if options is not None:
        control = dcc.Dropdown(
            id=f"input-{column}",
            options=options,
            placeholder=f"Select {label}",
            clearable=True,
        )
    else:
        control = dcc.Input(
            id=f"input-{column}",
            type="number",
            step=get_numeric_step(column),
            placeholder=f"Enter {label}",
            style={"width": "100%"},
        )
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Label(label, className="fw-semibold mb-2"),
                control,
                html.Small(helper_text, className="text-muted d-block mt-2"),
            ]),
            className="h-100 shadow-sm",
        ),
        md=6,
        lg=4,
        className="mb-3",
    )


def ordinal_category_maps():
    maps = {}
    for index, column in enumerate(ORDINAL_COLS):
        categories = feature_encoders["ordinal"].categories_[index]
        maps[column] = {category: float(code) for code, category in enumerate(categories)}
    return maps


ORDINAL_CATEGORY_MAPS = ordinal_category_maps()


def prepare_patient_features(raw_inputs: dict[str, object]):
    patient_raw = pd.DataFrame([{column: raw_inputs.get(column) for column in train_feature_cols}])
    patient_encoded = pd.DataFrame(index=[0], columns=train_feature_cols, dtype=float)

    for column in NUMERIC_COLS + BINARY_COLS:
        patient_encoded[column] = pd.to_numeric(patient_raw[column], errors="coerce")

    for column in ORDINAL_COLS:
        raw_value = patient_raw.at[0, column]
        if raw_value is None or raw_value == "":
            patient_encoded.at[0, column] = np.nan
        else:
            patient_encoded.at[0, column] = ORDINAL_CATEGORY_MAPS[column].get(raw_value, -1.0)

    for column in NOMINAL_COLS:
        raw_value = patient_raw.at[0, column]
        if raw_value is None or raw_value == "":
            patient_encoded.at[0, column] = np.nan
            continue
        encoder = feature_encoders["nominal"][column]
        try:
            patient_encoded.at[0, column] = float(encoder.transform([raw_value])[0])
        except ValueError:
            patient_encoded.at[0, column] = -1.0

    missing_columns = []
    for column in train_feature_cols:
        if pd.isna(patient_encoded.at[0, column]):
            patient_encoded.at[0, column] = train_fill_values[column]
            missing_columns.append(column)

    patient_encoded = patient_encoded[train_feature_cols].astype(float)
    patient_scaled = pd.DataFrame(
        scaler.transform(patient_encoded),
        columns=train_feature_cols,
    )
    return patient_raw, patient_encoded, patient_scaled, missing_columns


def compute_shap_values(explainer: shap.TreeExplainer, X_eval: pd.DataFrame):
    try:
        explanation = explainer(X_eval, check_additivity=False)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
    except TypeError:
        legacy_values = explainer.shap_values(X_eval, check_additivity=False)
        if isinstance(legacy_values, list):
            values = np.stack(legacy_values, axis=-1)
        else:
            values = np.asarray(legacy_values)
        base_values = np.asarray(explainer.expected_value)

    if values.ndim not in (2, 3):
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
    return values, base_values


def resolve_base_value(base_values: np.ndarray, sample_index: int, class_index: int) -> float:
    if np.isscalar(base_values):
        return float(base_values)
    if base_values.ndim == 0:
        return float(base_values.item())
    if base_values.ndim == 1:
        if len(base_values) > class_index:
            return float(base_values[class_index])
        return float(base_values[sample_index])
    if base_values.ndim == 2:
        return float(base_values[sample_index, class_index])
    raise ValueError(f"Unsupported base value shape: {base_values.shape}")


def get_shap_explainer():
    global shap_explainer
    if shap_explainer is None:
        shap_explainer = shap.TreeExplainer(
            PREDICTION_MODEL,
            data=shap_background,
        )
    return shap_explainer


def build_probability_figure(probabilities: np.ndarray, predicted_class: str) -> go.Figure:
    probability_df = pd.DataFrame(
        {
            "Class": target_encoder.classes_,
            "Probability": probabilities,
        }
    )
    probability_df["Type"] = np.where(
        probability_df["Class"] == predicted_class,
        "Predicted",
        "Other",
    )
    probability_df["ProbabilityLabel"] = probability_df["Probability"].map(lambda value: f"{value * 100:.1f}%")

    figure = px.bar(
        probability_df,
        x="Class",
        y="Probability",
        color="Type",
        text="ProbabilityLabel",
        color_discrete_map={"Predicted": "#0c7c59", "Other": "#9ec5ab"},
    )
    figure.update_layout(
        template="plotly_white",
        height=320,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Probability",
        xaxis_title="Diabetes Stage",
    )
    figure.update_traces(textposition="outside")
    return figure


def build_local_shap_outputs(patient_raw: pd.DataFrame, patient_encoded: pd.DataFrame):
    model = PREDICTION_MODEL
    predicted_index = int(model.predict(patient_encoded)[0])
    predicted_class = target_encoder.inverse_transform([predicted_index])[0]

    explainer = get_shap_explainer()
    shap_values, base_values = compute_shap_values(explainer, patient_encoded)

    if shap_values.ndim == 3:
        patient_values = shap_values[0, :, predicted_index]
    else:
        patient_values = shap_values[0]

    base_value = resolve_base_value(base_values, 0, predicted_index)

    contribution_df = pd.DataFrame(
        {
            "feature": train_feature_cols,
            "shap_value": patient_values,
            "raw_value": [patient_raw.at[0, column] for column in train_feature_cols],
        }
    )
    contribution_df["abs_shap_value"] = contribution_df["shap_value"].abs()
    contribution_df["display_feature"] = contribution_df["feature"].map(pretty_feature_name)
    contribution_df["direction"] = np.where(
        contribution_df["shap_value"] >= 0,
        "Pushes higher",
        "Pushes lower",
    )

    top_drivers = contribution_df.sort_values("abs_shap_value", ascending=False).head(10).copy()
    top_drivers = top_drivers.sort_values("shap_value")

    figure = px.bar(
        top_drivers,
        x="shap_value",
        y="display_feature",
        color="direction",
        orientation="h",
        text="shap_value",
        hover_data={
            "feature": False,
            "display_feature": False,
            "direction": True,
            "raw_value": True,
            "shap_value": ":.4f",
        },
        color_discrete_map={
            "Pushes higher": "#c0392b",
            "Pushes lower": "#2471a3",
        },
    )
    figure.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="SHAP value",
        yaxis_title="Feature",
        legend_title_text="Effect",
    )
    figure.update_traces(texttemplate="%{x:.3f}", textposition="outside")

    top_list = []
    for _, row in contribution_df.sort_values("abs_shap_value", ascending=False).head(3).iterrows():
        top_list.append(
            html.Li(
                f"{row['display_feature']}: {row['direction']} the predicted class "
                f"(input = {format_value(row['raw_value'])}, SHAP = {row['shap_value']:.3f})"
            )
        )

    summary = dbc.Card(
        dbc.CardBody(
            [
                html.H5(f"Local SHAP Explanation for {predicted_class}", className="card-title"),
                html.P(
                    f"Base value for the predicted class: {base_value:.3f}. "
                    "The chart below shows which patient features moved the prediction away from that baseline.",
                    className="mb-3",
                ),
                html.Ul(top_list, className="mb-0"),
            ]
        ),
        className="shadow-sm",
    )
    return predicted_class, summary, figure


def load_or_build_cluster_profiles():
    profile_path = DATA_DIR / "cluster_profiles.csv"
    if profile_path.exists():
        return pd.read_csv(profile_path).set_index("cluster")

    cluster_path = DATA_DIR / "train_clusters.csv"
    if not cluster_path.exists():
        return None

    cluster_labels = pd.read_csv(cluster_path).squeeze()
    profile_columns = [
        "physical_activity_minutes_per_week",
        "diet_score",
        "bmi",
        "glucose_fasting",
        "hba1c",
    ]
    available_columns = [column for column in profile_columns if column in X_train.columns]
    if not available_columns:
        return None

    cluster_frame = X_train[available_columns].copy()
    cluster_frame["cluster"] = cluster_labels.to_numpy()
    profiles = cluster_frame.groupby("cluster")[available_columns].mean().round(2)
    profiles["cluster_size"] = cluster_frame.groupby("cluster").size()
    profiles.reset_index().to_csv(profile_path, index=False)
    return profiles


cluster_profiles = load_or_build_cluster_profiles()


def build_recommendations(cluster_id: int):
    if cluster_profiles is None or cluster_id not in cluster_profiles.index:
        return dbc.Alert(
            "Cluster profiles are not available yet. Re-run src/train_models.py to generate them.",
            color="warning",
        )

    profile = cluster_profiles.loc[cluster_id]
    recommendation_lines = []

    if profile.get("physical_activity_minutes_per_week", 0) < max(150, cluster_baselines["physical_activity_minutes_per_week"]):
        recommendation_lines.append("Increase weekly physical activity toward at least 150 minutes.")
    if profile.get("diet_score", 0) < cluster_baselines["diet_score"]:
        recommendation_lines.append("Improve diet quality with more fiber-rich foods and fewer refined sugars.")
    if profile.get("bmi", 0) > cluster_baselines["bmi"]:
        recommendation_lines.append("Prioritize weight-management support and routine lifestyle follow-up.")
    if profile.get("glucose_fasting", 0) > cluster_baselines["glucose_fasting"]:
        recommendation_lines.append("Monitor fasting glucose trends closely and escalate clinical review when needed.")
    if profile.get("hba1c", 0) > cluster_baselines["hba1c"]:
        recommendation_lines.append("Schedule regular HbA1c follow-up and reinforce medication adherence if prescribed.")

    if not recommendation_lines:
        recommendation_lines.append("Current cluster profile is comparatively stable. Maintain the existing activity and diet pattern.")

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(f"Lifestyle Recommendations for Cluster {cluster_id}", className="card-title"),
                html.P(f"Average weekly activity: {profile.get('physical_activity_minutes_per_week', np.nan):.1f} minutes"),
                html.P(f"Average diet score: {profile.get('diet_score', np.nan):.1f}"),
                html.P(f"Average BMI: {profile.get('bmi', np.nan):.1f}"),
                html.Ul([html.Li(text) for text in recommendation_lines], className="mb-0"),
            ]
        ),
        className="shadow-sm",
    )


def build_global_shap_children():
    summary_file = ASSETS_DIR / f"{MODEL_ASSET_PREFIX}_shap_summary.png"
    bar_file = ASSETS_DIR / f"{MODEL_ASSET_PREFIX}_shap_bar.png"

    if not summary_file.exists() or not bar_file.exists():
        return dbc.Alert(
            "XGBoost SHAP assets are missing. Run src/shap_analysis.py to generate them.",
            color="warning",
        )

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Global SHAP Summary", className="card-title"),
                            html.Img(src=app.get_asset_url(summary_file.name), style={"width": "100%"}),
                        ]
                    ),
                    className="shadow-sm",
                ),
                lg=6,
                className="mb-3",
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Global Feature Importance", className="card-title"),
                            html.Img(src=app.get_asset_url(bar_file.name), style={"width": "100%"}),
                        ]
                    ),
                    className="shadow-sm",
                ),
                lg=6,
                className="mb-3",
            ),
        ]
    )


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=str(ASSETS_DIR),
)
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H2("Diabetes Risk Decision Support System", className="mb-2"),
                    html.P(
                        "Prediction, clustering, and SHAP explanations now share the same patient-preprocessing path.",
                        className="text-muted mb-4",
                    ),
                ]
            )
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Risk Prediction",
                    children=[
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Prediction Model", className="card-title"),
                                            html.P(
                                                "All diabetes-stage predictions and SHAP explanations in this app use the saved XGBoost classifier.",
                                                className="mb-0",
                                            ),
                                        ]
                                    ),
                                    className="shadow-sm mb-4",
                                )
                            )
                        ),
                        html.H4("Primary Inputs", className="mb-3"),
                        html.P(
                            "Any blank field, including the main ones below, will automatically use the baseline shown under that input.",
                            className="text-muted mb-3",
                        ),
                        dbc.Row([build_input_field(column) for column in PRIMARY_FEATURES]),
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        html.P(
                                            "These lower-impact fields are optional. Leave them blank to use the baseline shown below each field.",
                                            className="text-muted mb-3",
                                        ),
                                        dbc.Row([build_input_field(column) for column in SECONDARY_FEATURES]),
                                    ],
                                    title="Optional Advanced Inputs",
                                )
                            ],
                            start_collapsed=True,
                            className="mb-4",
                        ),
                        dbc.Button("Predict Risk", id="predict-btn", color="primary", className="mb-4"),
                        html.Div(id="prediction-output", className="mb-4"),
                        dcc.Graph(
                            id="probability-figure",
                            figure=build_empty_figure("Run a prediction to see class probabilities."),
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Patient Segmentation",
                    children=[
                        html.Div(className="mb-3"),
                        dbc.Button("Assign Cluster", id="cluster-btn", color="secondary", className="mb-4"),
                        html.Div(id="cluster-output", className="mb-4"),
                    ],
                ),
                dcc.Tab(
                    label="SHAP Insights",
                    children=[
                        html.Div(className="mb-3"),
                        html.H4("Model Driver Analysis"),
                        html.P(
                            "Global XGBoost plots come from src/shap_analysis.py. The local chart updates from the most recent patient prediction.",
                            className="text-muted mb-4",
                        ),
                        html.Div(build_global_shap_children(), id="global-shap-output", className="mb-4"),
                        html.Div(id="local-shap-summary", className="mb-4"),
                        dcc.Graph(
                            id="local-shap-figure",
                            figure=build_empty_figure("Submit a risk prediction to see local SHAP drivers."),
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Recommendations",
                    children=[
                        html.Div(className="mb-3"),
                        html.H4("Lifestyle Recommendations"),
                        html.Div(id="recommendations-output"),
                    ],
                ),
            ]
        ),
    ],
    fluid=True,
    className="py-4",
)
@app.callback(
    Output("prediction-output", "children"),
    Output("probability-figure", "figure"),
    Output("local-shap-summary", "children"),
    Output("local-shap-figure", "figure"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{column}", "value") for column in FEATURE_COLS],
)
def predict_risk(n_clicks, *values):
    if not n_clicks:
        return (
            "",
            build_empty_figure("Run a prediction to see class probabilities."),
            "",
            build_empty_figure("Submit a risk prediction to see local SHAP drivers."),
        )

    raw_inputs = dict(zip(FEATURE_COLS, values))
    patient_raw, patient_encoded, _, missing_columns = prepare_patient_features(raw_inputs)
    missing_primary = [column for column in missing_columns if column in PRIMARY_FEATURES]
    missing_secondary = [column for column in missing_columns if column in SECONDARY_FEATURES]

    model = PREDICTION_MODEL
    predicted_index = int(model.predict(patient_encoded)[0])
    predicted_class = target_encoder.inverse_transform([predicted_index])[0]

    proba_frame = pd.Series(model.predict_proba(patient_encoded)[0], index=model.classes_)
    aligned_probabilities = proba_frame.reindex(range(len(target_encoder.classes_)), fill_value=0.0).to_numpy()
    confidence = float(aligned_probabilities[predicted_index])

    prediction_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4(f"Predicted Diabetes Stage: {predicted_class}", className="card-title"),
                html.P(f"Model used: {MODEL_LABEL}"),
                html.P(f"Prediction confidence: {confidence * 100:.1f}%"),
                html.P(
                    f"Blank primary fields auto-filled from baseline: {len(missing_primary)} ({format_feature_list(missing_primary)})",
                    className="mb-1 text-muted",
                ),
                html.P(
                    f"Blank optional advanced fields auto-filled from baseline: {len(missing_secondary)} ({format_feature_list(missing_secondary)})",
                    className="mb-0 text-muted",
                ),
            ]
        ),
        className="shadow-sm",
    )

    probability_figure = build_probability_figure(aligned_probabilities, predicted_class)
    _, shap_summary, shap_figure = build_local_shap_outputs(patient_raw, patient_encoded)

    return prediction_card, probability_figure, shap_summary, shap_figure


@app.callback(
    Output("cluster-output", "children"),
    Output("recommendations-output", "children"),
    Input("cluster-btn", "n_clicks"),
    [State(f"input-{column}", "value") for column in FEATURE_COLS],
)
def assign_cluster(n_clicks, *values):
    if not n_clicks:
        return "", ""

    raw_inputs = dict(zip(FEATURE_COLS, values))
    _, _, patient_scaled, missing_columns = prepare_patient_features(raw_inputs)
    missing_primary = [column for column in missing_columns if column in PRIMARY_FEATURES]
    missing_secondary = [column for column in missing_columns if column in SECONDARY_FEATURES]
    cluster_id = int(kmeans_model.predict(patient_scaled)[0])

    cluster_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4(f"Assigned Cluster: {cluster_id}", className="card-title"),
                html.P(
                    f"Blank primary fields auto-filled from baseline: {len(missing_primary)} ({format_feature_list(missing_primary)})",
                    className="mb-1 text-muted",
                ),
                html.P(
                    f"Blank optional advanced fields auto-filled from baseline: {len(missing_secondary)} ({format_feature_list(missing_secondary)})",
                    className="mb-0 text-muted",
                ),
            ]
        ),
        className="shadow-sm",
    )

    recommendations = build_recommendations(cluster_id)
    return cluster_card, recommendations


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )