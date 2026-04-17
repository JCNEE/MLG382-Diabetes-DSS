import importlib.util
import os
from pathlib import Path
import pickle
from typing import Any, Protocol, cast

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
APP_DESIGN_DIR = PROJECT_ROOT / "app design"
WEB_DESIGN_FILE = APP_DESIGN_DIR / "web_design.py"


class WebDesignModule(Protocol):
    def build_index_string(self, css_text: str) -> str: ...
    def build_input_card(self, label: str, helper_text: str, control: Any): ...
    def build_layout(
        self,
        *,
        primary_inputs: Any,
        secondary_inputs: Any,
        global_shap_children: Any,
        model_label: str,
        feature_count: int,
        training_rows: int,
        probability_figure: Any,
        local_shap_figure: Any,
    ): ...
    def load_theme_css(self) -> str: ...


def load_web_design_module() -> WebDesignModule:
    spec = importlib.util.spec_from_file_location("web_design", WEB_DESIGN_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load web_design module from {WEB_DESIGN_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(WebDesignModule, module)


web_design = load_web_design_module()
build_index_string = web_design.build_index_string
build_input_card = web_design.build_input_card
build_layout = web_design.build_layout
load_theme_css = web_design.load_theme_css

DEFAULT_MODEL_NAME = "xgboost"
MODEL_CONFIGS = {
    "xgboost": {
        "label": "XGBoost",
        "model_file": "xgboost.pkl",
        "asset_prefix": "XGBoost",
        "shap_csv": "xgboost_shap_feature_importance.csv",
    },
    "decision_tree": {
        "label": "Decision Tree",
        "model_file": "decision_tree.pkl",
        "asset_prefix": "DecisionTree",
        "shap_csv": "decision_tree_shap_feature_importance.csv",
    },
    "random_forest": {
        "label": "Random Forest",
        "model_file": "random_forest.pkl",
        "asset_prefix": "RandomForest",
        "shap_csv": "random_forest_shap_feature_importance.csv",
    },
}
MODEL_OPTIONS = [
    {"label": model_config["label"], "value": model_name}
    for model_name, model_config in MODEL_CONFIGS.items()
]

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
GLOBAL_IMPORTANCE_MAX_FEATURES = 12


def load_pickle(path: Path):
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def normalise_model_name(model_name: str | None) -> str:
    if model_name in MODEL_CONFIGS:
        return model_name
    return DEFAULT_MODEL_NAME


def get_model_config(model_name: str | None) -> dict[str, str]:
    return MODEL_CONFIGS[normalise_model_name(model_name)]


def get_model_label(model_name: str | None) -> str:
    return get_model_config(model_name)["label"]


def get_model_asset_prefix(model_name: str | None) -> str:
    return get_model_config(model_name)["asset_prefix"]


def get_model_shap_path(model_name: str | None) -> Path:
    return DATA_DIR / get_model_config(model_name)["shap_csv"]


def get_model_file(model_name: str | None) -> Path:
    return MODELS_DIR / get_model_config(model_name)["model_file"]


def load_prediction_model(model_name: str):
    model_label = get_model_label(model_name)
    model_path = get_model_file(model_name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_label} model. Run src/train_models.py first."
        )
    return load_pickle(model_path)


PREDICTION_MODELS = {
    model_name: load_prediction_model(model_name)
    for model_name in MODEL_CONFIGS
}

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
shap_explainers: dict[str, shap.TreeExplainer] = {}
cluster_baselines = X_train[
    [
        "physical_activity_minutes_per_week",
        "diet_score",
        "bmi",
        "glucose_fasting",
        "hba1c",
    ]
].median().to_dict()


def load_global_shap_importance(model_name: str):
    shap_path = get_model_shap_path(model_name)
    if not shap_path.exists():
        return None

    shap_frame = pd.read_csv(shap_path)
    if "feature" not in shap_frame.columns:
        return None

    return shap_frame


GLOBAL_SHAP_IMPORTANCE_CACHE = {
    model_name: load_global_shap_importance(model_name)
    for model_name in MODEL_CONFIGS
}


def get_global_shap_importance(model_name: str | None):
    return GLOBAL_SHAP_IMPORTANCE_CACHE.get(normalise_model_name(model_name))


def build_global_importance_options(model_name: str | None):
    global_shap_importance = get_global_shap_importance(model_name)
    if global_shap_importance is None:
        return []

    options = []
    if "mean_abs_shap_all_classes" in global_shap_importance.columns:
        options.append(
            {
                "label": "All diabetes stages",
                "value": "mean_abs_shap_all_classes",
            }
        )

    for class_name in target_encoder.classes_:
        column_name = f"mean_abs_shap_{class_name}"
        if column_name in global_shap_importance.columns:
            options.append({"label": class_name, "value": column_name})

    return options


def get_default_global_importance_view(model_name: str | None):
    options = build_global_importance_options(model_name)
    return options[0]["value"] if options else None


def load_primary_features():
    default_global_importance = get_global_shap_importance(DEFAULT_MODEL_NAME)
    if default_global_importance is not None:
        ranked = [
            feature for feature in default_global_importance["feature"].tolist()
            if feature in FEATURE_COLS
        ]
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
    figure = apply_chart_theme(figure, height=320)
    figure.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
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
                "font": {"size": 14, "color": "#5f7268"},
            }
        ],
    )
    return figure


def apply_chart_theme(figure: go.Figure, height: int) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": "#183128"},
        margin=dict(l=20, r=20, t=32, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
    )
    figure.update_xaxes(gridcolor="rgba(24, 49, 40, 0.08)", zerolinecolor="rgba(24, 49, 40, 0.08)")
    figure.update_yaxes(gridcolor="rgba(24, 49, 40, 0.08)", zerolinecolor="rgba(24, 49, 40, 0.08)")
    return figure


def describe_global_importance_view(selected_column: str | None) -> str:
    if selected_column == "mean_abs_shap_all_classes":
        return "All diabetes stages"

    prefix = "mean_abs_shap_"
    if selected_column and selected_column.startswith(prefix):
        return selected_column[len(prefix):]

    return "All diabetes stages"


def build_global_importance_figure(
    selected_column: str | None,
    model_name: str | None = None,
) -> go.Figure:
    resolved_model_name = normalise_model_name(model_name)
    model_label = get_model_label(resolved_model_name)
    global_shap_importance = get_global_shap_importance(resolved_model_name)

    if global_shap_importance is None:
        return build_empty_figure(
            f"Run src/shap_analysis.py --model {resolved_model_name} to load global feature importance for {model_label}."
        )

    metric_column = selected_column
    if metric_column not in global_shap_importance.columns:
        metric_column = get_default_global_importance_view(resolved_model_name)

    if metric_column is None:
        return build_empty_figure(
            f"Class-level SHAP columns are not available in {get_model_shap_path(resolved_model_name).name}."
        )

    importance_frame = global_shap_importance[["feature", metric_column]].copy()
    importance_frame[metric_column] = pd.to_numeric(
        importance_frame[metric_column], errors="coerce"
    ).fillna(0.0)
    importance_frame = importance_frame.rename(columns={metric_column: "importance"})

    if importance_frame.empty:
        return build_empty_figure("No SHAP feature-importance rows are available.")

    importance_frame["display_feature"] = importance_frame["feature"].map(pretty_feature_name)
    importance_frame = importance_frame.sort_values("importance", ascending=False).head(
        GLOBAL_IMPORTANCE_MAX_FEATURES
    )
    importance_frame = importance_frame.iloc[::-1]

    figure = px.bar(
        importance_frame,
        x="importance",
        y="display_feature",
        orientation="h",
        hover_data={
            "feature": True,
            "display_feature": False,
            "importance": ":.4f",
        },
        color_discrete_sequence=["#2f8f68"],
    )
    figure = apply_chart_theme(figure, height=360)
    figure.update_layout(
        xaxis_title="Mean |SHAP|",
        yaxis_title=None,
        showlegend=False,
        margin=dict(l=20, r=20, t=12, b=24),
    )
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    figure.update_traces(
        marker_line_width=0,
        hovertemplate="%{y}<br>Mean |SHAP|: %{x:.3f}<extra></extra>",
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
            className="wellness-dropdown",
        )
    else:
        control = dcc.Input(
            id=f"input-{column}",
            type="number",
            step=get_numeric_step(column),
            placeholder=f"Enter {label}",
            className="wellness-control",
        )
    return build_input_card(label=label, helper_text=helper_text, control=control)


def build_model_selector():
    return dcc.Dropdown(
        id="model-selector",
        options=MODEL_OPTIONS,
        value=DEFAULT_MODEL_NAME,
        clearable=False,
        className="wellness-dropdown model-selector-dropdown",
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


def get_prediction_model(model_name: str | None):
    return PREDICTION_MODELS[normalise_model_name(model_name)]


def get_shap_explainer(model_name: str | None):
    resolved_model_name = normalise_model_name(model_name)
    if resolved_model_name not in shap_explainers:
        shap_explainers[resolved_model_name] = shap.TreeExplainer(
            get_prediction_model(resolved_model_name),
            data=shap_background,
        )
    return shap_explainers[resolved_model_name]


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
        color_discrete_map={"Predicted": "#2f8f68", "Other": "#cbdcd0"},
    )
    figure = apply_chart_theme(figure, height=320)
    figure.update_layout(showlegend=False, yaxis_title="Probability", xaxis_title="Diabetes Stage")
    figure.update_yaxes(tickformat=".0%")
    figure.update_traces(textposition="outside", marker_line_width=0)
    return figure


def build_local_shap_outputs(
    patient_raw: pd.DataFrame,
    patient_encoded: pd.DataFrame,
    model_name: str | None,
):
    resolved_model_name = normalise_model_name(model_name)
    model_label = get_model_label(resolved_model_name)
    model = get_prediction_model(resolved_model_name)
    predicted_index = int(model.predict(patient_encoded)[0])
    predicted_class = target_encoder.inverse_transform([predicted_index])[0]
    probability_frame = pd.Series(model.predict_proba(patient_encoded)[0], index=model.classes_)
    predicted_probability = float(probability_frame.get(predicted_index, 0.0))

    explainer = get_shap_explainer(resolved_model_name)
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
        "Supports predicted stage",
        "Pulls away from predicted stage",
    )
    total_abs_shap = float(contribution_df["abs_shap_value"].sum())
    contribution_df["impact_share"] = (
        contribution_df["abs_shap_value"] / total_abs_shap if total_abs_shap else 0.0
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
            "Supports predicted stage": "#2f8f68",
            "Pulls away from predicted stage": "#c9774c",
        },
    )
    figure = apply_chart_theme(figure, height=420)
    figure.update_layout(
        xaxis_title="SHAP value (effect on predicted stage score)",
        yaxis_title="Feature",
        legend_title_text="Contribution",
    )
    figure.update_traces(texttemplate="%{x:.3f}", textposition="outside")

    supporting_features = contribution_df[contribution_df["shap_value"] > 0].sort_values(
        "abs_shap_value", ascending=False
    ).head(3)
    opposing_features = contribution_df[contribution_df["shap_value"] < 0].sort_values(
        "abs_shap_value", ascending=False
    ).head(3)

    def build_driver_items(driver_frame: pd.DataFrame, *, supports_stage: bool):
        items = []
        for _, row in driver_frame.iterrows():
            if supports_stage:
                explanation_text = f"increased model support for {predicted_class}"
            else:
                explanation_text = f"pulled the model away from {predicted_class}"

            items.append(
                html.Li(
                    f"{row['display_feature']}: input = {format_value(row['raw_value'])}; "
                    f"this {explanation_text} "
                    f"(SHAP = {row['shap_value']:.3f}, impact share = {row['impact_share'] * 100:.1f}%)."
                )
            )
        return items

    supporting_items = build_driver_items(supporting_features, supports_stage=True)
    opposing_items = build_driver_items(opposing_features, supports_stage=False)

    summary = dbc.Card(
        dbc.CardBody(
            [
                html.H5(f"Why {model_label} Predicted {predicted_class}", className="card-title"),
                html.P(
                    f"{model_label} assigned {predicted_probability * 100:.1f}% probability to {predicted_class}. "
                    f"Local SHAP starts from the class base score ({base_value:.3f}), which is the model's raw starting score for this stage before patient-specific features are applied.",
                    className="mb-2",
                ),
                html.P(
                    "Positive SHAP values support the predicted stage, while negative values pull the model away from it. Larger absolute SHAP values mean the feature had a stronger effect on the final decision. The chart below ranks the ten strongest patient-specific drivers.",
                    className="mb-3",
                ),
                html.Div("Strongest Features Supporting This Stage", className="support-label"),
                html.Ul(supporting_items, className="mb-3") if supporting_items else html.P(
                    "No major patient features are strongly increasing the model's support for this stage.",
                    className="mb-3",
                ),
                html.Div("Strongest Features Pulling Against This Stage", className="support-label"),
                html.Ul(opposing_items, className="mb-0") if opposing_items else html.P(
                    "No major patient features are strongly pulling the model away from this stage.",
                    className="mb-0",
                ),
            ]
        ),
        className="analysis-card h-100",
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

CLUSTER_SEGMENT_LABELS = {
    "healthy": "Healthy Patient Cluster",
    "elevated_glucose": "Elevated Glucose Patient Cluster",
    "unhealthy": "Unhealthy Patient Cluster",
}

CLUSTER_SEGMENT_SUMMARIES = {
    "healthy": (
        "This segment represents the most stable overall profile in the training data. "
        "Patients in this group tend to be more active, slightly better on diet quality, "
        "leaner on average, and lower on glucose markers than the typical patient."
    ),
    "elevated_glucose": (
        "This segment is defined mainly by poorer glucose control. Patients here do not "
        "necessarily have the highest BMI, but their fasting glucose and HbA1c values are "
        "the highest of the three groups, so blood sugar management is the main concern."
    ),
    "unhealthy": (
        "This segment reflects the broadest lifestyle and metabolic burden in the training data. "
        "Patients in this group tend to have the highest BMI, the weakest diet score, and glucose "
        "markers that remain above the typical training profile."
    ),
}


def build_cluster_segment_keys():
    if cluster_profiles is None or cluster_profiles.empty:
        return {}

    required_columns = [
        "physical_activity_minutes_per_week",
        "diet_score",
        "bmi",
        "glucose_fasting",
        "hba1c",
    ]
    if any(column not in cluster_profiles.columns for column in required_columns):
        return {}

    profiles = cluster_profiles[required_columns].apply(pd.to_numeric, errors="coerce")
    segment_keys: dict[int, str] = {}

    overall_health_rank = (
        profiles["physical_activity_minutes_per_week"].rank(method="dense", ascending=False)
        + profiles["diet_score"].rank(method="dense", ascending=False)
        + profiles["bmi"].rank(method="dense", ascending=True)
        + profiles["glucose_fasting"].rank(method="dense", ascending=True)
        + profiles["hba1c"].rank(method="dense", ascending=True)
    )
    healthiest_cluster = int(overall_health_rank.idxmin())
    segment_keys[healthiest_cluster] = "healthy"

    remaining_clusters = [cluster_id for cluster_id in profiles.index if int(cluster_id) not in segment_keys]
    if remaining_clusters:
        remaining_profiles = profiles.loc[remaining_clusters]
        glucose_risk_rank = (
            remaining_profiles["glucose_fasting"].rank(method="dense", ascending=False)
            + remaining_profiles["hba1c"].rank(method="dense", ascending=False)
        )
        highest_glucose_cluster = int(glucose_risk_rank.idxmin())
        segment_keys[highest_glucose_cluster] = "elevated_glucose"

    for cluster_id in profiles.index:
        cluster_key = int(cluster_id)
        if cluster_key in segment_keys:
            continue
        segment_keys[cluster_key] = "unhealthy"

    return segment_keys


CLUSTER_SEGMENT_KEYS = build_cluster_segment_keys()


def get_cluster_segment_key(cluster_id: int) -> str | None:
    return CLUSTER_SEGMENT_KEYS.get(cluster_id)


def get_cluster_display_name(cluster_id: int) -> str:
    segment_key = get_cluster_segment_key(cluster_id)
    if segment_key is None:
        return f"Cluster {cluster_id}"
    return CLUSTER_SEGMENT_LABELS.get(segment_key, f"Cluster {cluster_id}")


def describe_cluster_position(value: float, baseline: float, *, higher_is_healthier: bool) -> str:
    if pd.isna(value) or pd.isna(baseline):
        return "relative position not available"
    if np.isclose(value, baseline):
        return "around the training median"
    if higher_is_healthier:
        return "above the training median" if value > baseline else "below the training median"
    return "below the training median" if value < baseline else "above the training median"


def build_cluster_meaning(cluster_id: int):
    if cluster_profiles is None or cluster_id not in cluster_profiles.index:
        return None

    profile = cluster_profiles.loc[cluster_id]
    segment_key = get_cluster_segment_key(cluster_id)
    summary = CLUSTER_SEGMENT_SUMMARIES.get(
        segment_key,
        "This cluster groups patients who share a similar overall lifestyle and metabolic pattern in the training data.",
    )
    cluster_size = int(profile.get("cluster_size", 0))

    details = [
        "Being assigned to this segment means the current patient looks most similar to this group in the training data. It is a similarity profile, not a diagnosis by itself.",
        f"This segment contains {cluster_size:,} training patients with a similar overall pattern.",
        (
            f"Average weekly activity is {profile.get('physical_activity_minutes_per_week', np.nan):.1f} minutes, "
            f"which is {describe_cluster_position(profile.get('physical_activity_minutes_per_week', np.nan), cluster_baselines['physical_activity_minutes_per_week'], higher_is_healthier=True)} "
            f"compared with the training median of {cluster_baselines['physical_activity_minutes_per_week']:.1f}."
        ),
        (
            f"Average diet score is {profile.get('diet_score', np.nan):.2f}, "
            f"which is {describe_cluster_position(profile.get('diet_score', np.nan), cluster_baselines['diet_score'], higher_is_healthier=True)} "
            f"compared with the training median of {cluster_baselines['diet_score']:.2f}."
        ),
        (
            f"Average BMI is {profile.get('bmi', np.nan):.2f}, "
            f"which is {describe_cluster_position(profile.get('bmi', np.nan), cluster_baselines['bmi'], higher_is_healthier=False)} "
            f"compared with the training median of {cluster_baselines['bmi']:.2f}."
        ),
        (
            f"Average fasting glucose is {profile.get('glucose_fasting', np.nan):.2f}, "
            f"which is {describe_cluster_position(profile.get('glucose_fasting', np.nan), cluster_baselines['glucose_fasting'], higher_is_healthier=False)} "
            f"compared with the training median of {cluster_baselines['glucose_fasting']:.2f}."
        ),
        (
            f"Average HbA1c is {profile.get('hba1c', np.nan):.2f}, "
            f"which is {describe_cluster_position(profile.get('hba1c', np.nan), cluster_baselines['hba1c'], higher_is_healthier=False)} "
            f"compared with the training median of {cluster_baselines['hba1c']:.2f}."
        ),
    ]

    return {
        "summary": summary,
        "details": details,
    }


def build_stage_meaning_items(predicted_class: str | None):
    if predicted_class == "No Diabetes":
        return [
            "The current inputs are closer to a non-diabetes pattern than to the diabetes stages in this model.",
            "This is the lowest-risk result in the tool, but it is not a guarantee that diabetes cannot develop later.",
        ]

    if predicted_class == "Pre-Diabetes":
        return [
            "The current inputs suggest glucose regulation may be starting to drift outside the healthy range.",
            "This is a warning stage where early lifestyle change and clinical follow-up can still slow or prevent progression.",
        ]

    if predicted_class == "Type 2":
        return [
            "The current inputs are more consistent with a Type 2 diabetes pattern than the other stages in this model.",
            "This usually reflects sustained glucose-control strain and should be treated as a prompt for clinical confirmation rather than a standalone diagnosis.",
        ]

    if predicted_class == "Type 1":
        return [
            "The current inputs are most consistent with a Type 1 diabetes pattern in this model.",
            "A new or unexpected Type 1 pattern should be treated more urgently than a routine watch-and-wait result because Type 1 diabetes can worsen quickly.",
        ]

    if predicted_class == "Gestational":
        return [
            "The current inputs are most consistent with a gestational diabetes pattern in this model.",
            "Gestational diabetes needs pregnancy-specific review because both maternal and fetal health can be affected if glucose remains uncontrolled.",
        ]

    return [
        "The predicted stage shows which diabetes pattern the model considers the best fit for the current inputs.",
        "This result should be reviewed alongside formal tests, symptoms, and clinical history rather than used on its own.",
    ]


def build_combined_meaning_note(predicted_class: str | None, segment_key: str | None):
    if predicted_class == "No Diabetes":
        if segment_key == "healthy":
            return "Both the stage result and the segment suggest a relatively stable overall profile, so the main goal is to maintain healthy habits and keep screening routine."
        if segment_key == "elevated_glucose":
            return "The stage result is not diabetes, but the segment still suggests some glucose pressure compared with the healthiest group, so prevention should stay active."
        if segment_key == "unhealthy":
            return "The stage result is less concerning than diabetes, but the segment still suggests lifestyle or metabolic strain that can raise future risk."

    if predicted_class == "Pre-Diabetes":
        if segment_key == "healthy":
            return "This combination suggests glucose may be rising even though the broader lifestyle profile is comparatively stable, so early action still matters."
        if segment_key == "elevated_glucose":
            return "Both the stage result and the segment point toward glucose burden, so this is a strong prevention-and-follow-up signal."
        if segment_key == "unhealthy":
            return "This combination suggests both rising glucose risk and a broader lifestyle or metabolic burden, so prevention should be structured rather than casual."

    if predicted_class == "Type 2":
        if segment_key == "healthy":
            return "The diabetes-stage result is more concerning than the broader segment profile, which can happen when glucose markers are carrying much of the signal."
        if segment_key == "elevated_glucose":
            return "Both the stage result and the segment point strongly toward poor glucose control, so medical review should not be delayed."
        if segment_key == "unhealthy":
            return "Both the stage result and the segment suggest a broader metabolic burden rather than one isolated issue, so follow-up should be comprehensive."

    if predicted_class == "Type 1":
        return "For a Type 1 pattern, the predicted stage matters more clinically than the segment label. Use the segment as lifestyle context, but let urgency and follow-up be driven by the stage result."

    if predicted_class == "Gestational":
        return "For a gestational diabetes pattern, the predicted stage matters more clinically than the segment label. Use the segment as background context, but let pregnancy-specific care drive the plan."

    return "The stage prediction estimates which diabetes pattern fits best, while the segment shows what broader patient profile the person resembles. Together they help focus follow-up."


def get_entered_numeric_value(
    patient_raw: pd.DataFrame | None,
    missing_columns: list[str] | None,
    column: str,
) -> float | None:
    if patient_raw is None or column not in patient_raw.columns:
        return None
    if missing_columns and column in missing_columns:
        return None

    value = patient_raw.at[0, column]
    if value is None or value == "":
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric_value):
        return None
    return numeric_value


def get_entered_text_value(
    patient_raw: pd.DataFrame | None,
    missing_columns: list[str] | None,
    column: str,
) -> str | None:
    if patient_raw is None or column not in patient_raw.columns:
        return None
    if missing_columns and column in missing_columns:
        return None

    value = patient_raw.at[0, column]
    if value is None or value == "":
        return None
    return str(value)


def build_patient_specific_actions(
    patient_raw: pd.DataFrame | None,
    missing_columns: list[str] | None,
    predicted_class: str | None,
):
    actions = []

    activity_value = get_entered_numeric_value(patient_raw, missing_columns, "physical_activity_minutes_per_week")
    if activity_value is not None:
        if activity_value < 90:
            actions.append(
                f"Physical activity is currently {activity_value:.0f} minutes per week, which is well below the usual 150-minute target; increasing movement is one of the clearest priorities."
            )
        elif activity_value < 150:
            actions.append(
                f"Physical activity is {activity_value:.0f} minutes per week, so increasing it toward 150 minutes per week would better support glucose control and weight management."
            )

    diet_value = get_entered_numeric_value(patient_raw, missing_columns, "diet_score")
    if diet_value is not None and diet_value < cluster_baselines["diet_score"]:
        actions.append(
            f"The diet score is {diet_value:.2f}, which is below the training median of {cluster_baselines['diet_score']:.2f}; diet quality should improve, especially around refined carbohydrates, sugary drinks, and fiber intake."
        )

    bmi_value = get_entered_numeric_value(patient_raw, missing_columns, "bmi")
    if bmi_value is not None:
        if bmi_value >= 30:
            actions.append(
                f"BMI is {bmi_value:.2f}, which suggests obesity-range weight burden; structured weight-management support could have a meaningful effect on glucose and cardiovascular risk."
            )
        elif bmi_value >= 25:
            actions.append(
                f"BMI is {bmi_value:.2f}, which suggests excess weight may be contributing to risk; even modest weight reduction could help improve metabolic control."
            )

    fasting_glucose = get_entered_numeric_value(patient_raw, missing_columns, "glucose_fasting")
    if fasting_glucose is not None:
        if fasting_glucose >= 126:
            actions.append(
                f"Fasting glucose is {fasting_glucose:.2f}, which is already in a clearly raised range and should be reviewed clinically rather than monitored casually."
            )
        elif fasting_glucose >= 100:
            actions.append(
                f"Fasting glucose is {fasting_glucose:.2f}, which is above the usual healthy range and supports closer follow-up of glucose markers."
            )

    postprandial_glucose = get_entered_numeric_value(patient_raw, missing_columns, "glucose_postprandial")
    if postprandial_glucose is not None:
        if postprandial_glucose >= 200:
            actions.append(
                f"Post-meal glucose is {postprandial_glucose:.2f}, which is markedly raised and strengthens the case for prompt clinical review."
            )
        elif postprandial_glucose >= 140:
            actions.append(
                f"Post-meal glucose is {postprandial_glucose:.2f}, which suggests meal-related glucose control may need attention."
            )

    hba1c_value = get_entered_numeric_value(patient_raw, missing_columns, "hba1c")
    if hba1c_value is not None:
        if hba1c_value >= 6.5:
            actions.append(
                f"HbA1c is {hba1c_value:.2f}, which is in a diabetic-range marker and supports prompt clinical review."
            )
        elif hba1c_value >= 5.7:
            actions.append(
                f"HbA1c is {hba1c_value:.2f}, which is above the usual healthy range and suggests glucose control should be followed more closely."
            )

    systolic_bp = get_entered_numeric_value(patient_raw, missing_columns, "systolic_bp")
    diastolic_bp = get_entered_numeric_value(patient_raw, missing_columns, "diastolic_bp")
    if systolic_bp is not None or diastolic_bp is not None:
        systolic_text = f"{systolic_bp:.0f}" if systolic_bp is not None else "?"
        diastolic_text = f"{diastolic_bp:.0f}" if diastolic_bp is not None else "?"
        if (systolic_bp is not None and systolic_bp >= 140) or (diastolic_bp is not None and diastolic_bp >= 90):
            actions.append(
                f"Blood pressure is {systolic_text}/{diastolic_text}, which is clearly raised and should be managed alongside diabetes risk because cardiovascular risk compounds quickly."
            )
        elif (systolic_bp is not None and systolic_bp >= 130) or (diastolic_bp is not None and diastolic_bp >= 80):
            actions.append(
                f"Blood pressure is {systolic_text}/{diastolic_text}, which is above ideal and worth addressing as part of the overall metabolic-risk plan."
            )

    ldl_value = get_entered_numeric_value(patient_raw, missing_columns, "ldl_cholesterol")
    triglycerides_value = get_entered_numeric_value(patient_raw, missing_columns, "triglycerides")
    if ldl_value is not None and ldl_value >= 130:
        actions.append(
            f"LDL cholesterol is {ldl_value:.2f}, so lipid management should be part of the follow-up plan, especially if diabetes risk is already elevated."
        )
    if triglycerides_value is not None and triglycerides_value >= 150:
        actions.append(
            f"Triglycerides are {triglycerides_value:.2f}, which suggests additional metabolic strain and strengthens the case for nutrition-focused follow-up."
        )

    sleep_value = get_entered_numeric_value(patient_raw, missing_columns, "sleep_hours_per_day")
    if sleep_value is not None and (sleep_value < 7 or sleep_value > 9):
        actions.append(
            f"Sleep duration is {sleep_value:.1f} hours per day, so sleep pattern may be adding metabolic strain; aiming for a steadier 7 to 9 hours would be more supportive."
        )

    screen_time = get_entered_numeric_value(patient_raw, missing_columns, "screen_time_hours_per_day")
    if screen_time is not None and screen_time > 6:
        actions.append(
            f"Screen time is {screen_time:.1f} hours per day, which suggests prolonged sedentary time; adding regular movement breaks through the day would help."
        )

    alcohol_value = get_entered_numeric_value(patient_raw, missing_columns, "alcohol_consumption_per_week")
    if alcohol_value is not None and alcohol_value > 14:
        actions.append(
            f"Alcohol intake is {alcohol_value:.1f} drinks per week, so reducing intake may help with glucose control, blood pressure, and weight management."
        )

    smoking_status = get_entered_text_value(patient_raw, missing_columns, "smoking_status")
    if smoking_status and "current" in smoking_status.lower():
        actions.append(
            "Current smoking was reported, so smoking cessation support should be part of the plan because it increases cardiovascular and metabolic risk."
        )

    family_history = get_entered_numeric_value(patient_raw, missing_columns, "family_history_diabetes")
    if family_history is not None and family_history >= 1:
        actions.append(
            "Family history of diabetes was reported, which increases baseline risk and supports more proactive monitoring."
        )

    hypertension_history = get_entered_numeric_value(patient_raw, missing_columns, "hypertension_history")
    if hypertension_history is not None and hypertension_history >= 1:
        actions.append(
            "A history of hypertension was reported, so blood pressure control should remain part of the diabetes prevention or management plan."
        )

    cardiovascular_history = get_entered_numeric_value(patient_raw, missing_columns, "cardiovascular_history")
    if cardiovascular_history is not None and cardiovascular_history >= 1:
        actions.append(
            "A history of cardiovascular disease was reported, which raises the urgency of follow-up because diabetes and cardiovascular risk reinforce each other."
        )

    if not actions:
        if predicted_class == "No Diabetes":
            actions.append(
                "The values entered do not show one strong patient-specific risk flag, so the main priority is maintaining healthy habits and staying consistent with routine screening."
            )
        else:
            actions.append(
                "The values entered do not point to one dominant lifestyle issue, so the safest approach is to follow the stage-specific guidance and keep monitoring consistent."
            )

    return actions


def build_personalisation_note(missing_columns: list[str] | None):
    if missing_columns is None:
        return "Patient-specific priorities use the values entered into the form."

    entered_count = len(FEATURE_COLS) - len(missing_columns)
    if not missing_columns:
        return f"Patient-specific priorities use all {entered_count} entered inputs from the form."

    return (
        f"Patient-specific priorities use the {entered_count} values entered into the form. "
        "Blank fields were excluded from this personalised section."
    )


def build_stage_guidance(predicted_class: str | None):
    if predicted_class == "No Diabetes":
        return [
            "Treat this as a prevention result, not as permission to stop monitoring risk factors.",
            "The main priorities are maintaining physical activity, a stable weight, good sleep, and a diet that does not gradually push glucose higher over time.",
            "If family history, blood pressure, or glucose markers are borderline, keep follow-up proactive even though the stage result is the lowest-risk one.",
        ]

    if predicted_class == "Pre-Diabetes":
        return [
            "This is the stage where lifestyle change has the best chance to delay or prevent progression to diabetes.",
            "The most useful priorities are regular weekly activity, lower intake of refined carbohydrates and sugary drinks, weight reduction if needed, and steady follow-up of glucose markers.",
            "Avoid a passive wait-and-see approach; this stage is best handled with a specific prevention plan.",
        ]

    if predicted_class == "Type 2":
        return [
            "Type 2 guidance should combine medical review with lifestyle change rather than relying on lifestyle alone.",
            "The practical priorities are glucose control, nutrition quality, regular activity, weight management if needed, and control of blood pressure and cholesterol risk.",
            "Long-term complication prevention matters here, so eye, kidney, foot, and cardiovascular follow-up should be part of the plan.",
        ]

    if predicted_class == "Type 1":
        return [
            "Type 1 diabetes is not mainly managed through lifestyle alone; insulin education and close clinical support are central.",
            "The practical priorities are glucose monitoring, hypoglycaemia safety, sick-day planning, and clear action steps for rising glucose or ketones.",
            "Lifestyle habits still matter, but they support treatment rather than replacing it.",
        ]

    if predicted_class == "Gestational":
        return [
            "Gestational diabetes guidance needs to be pregnancy-specific rather than treated like standard Type 2 advice.",
            "The practical priorities are glucose monitoring, meal planning that fits pregnancy care goals, and close coordination with the obstetric team.",
            "Follow-up should continue after delivery as well, because future diabetes risk can remain higher even when pregnancy ends.",
        ]

    return [
        "Use the predicted stage as a guide to urgency and follow-up rather than as a final diagnosis.",
        "The safest approach is structured clinical review plus steady improvement in lifestyle risk factors.",
    ]


def build_stage_next_steps(predicted_class: str | None):
    if predicted_class == "No Diabetes":
        return [
            "Maintain the current healthy pattern and continue routine diabetes screening during regular primary care visits.",
            "If there is family history, raised blood pressure, or increasing glucose markers, ask for repeat fasting glucose or HbA1c testing rather than waiting for symptoms.",
            "Use this result as a prevention window: keep activity up, protect sleep, and avoid gradual weight gain over time.",
        ]

    if predicted_class == "Pre-Diabetes":
        return [
            "Arrange a primary care follow-up to confirm the risk pattern with formal lab review and repeat HbA1c or fasting glucose testing.",
            "Start a structured prevention plan now, especially regular activity, carbohydrate quality improvement, and weight reduction if needed.",
            "Ask whether referral to a dietitian, diabetes prevention programme, or more frequent monitoring is appropriate.",
        ]

    if predicted_class == "Type 2":
        return [
            "Arrange prompt clinical review to confirm the diagnosis, review glucose and HbA1c results, and decide whether medication is needed.",
            "Discuss a full diabetes care plan, including blood pressure, cholesterol, kidney monitoring, eye screening, foot checks, and self-monitoring guidance.",
            "Begin lifestyle changes immediately while the medical plan is being confirmed, especially around activity, nutrition, and weight management.",
        ]

    if predicted_class == "Type 1":
        return [
            "If this is a new or unexpected result, seek urgent medical review because Type 1 diabetes needs rapid clinical assessment and management.",
            "If Type 1 has already been diagnosed, the next steps are close coordination with a diabetes specialist team, insulin education, glucose monitoring, and sick-day planning.",
            "Ask for guidance on ketone monitoring, hypoglycaemia management, and when to seek urgent care if symptoms worsen.",
        ]

    if predicted_class == "Gestational":
        return [
            "Contact the obstetric and diabetes care team promptly because gestational diabetes needs close pregnancy-specific follow-up.",
            "Review blood glucose monitoring targets, meal planning, and whether medication or insulin is needed during pregnancy.",
            "Plan postpartum glucose follow-up as well, because diabetes risk can remain elevated after delivery.",
        ]

    return [
        "Review the result with a clinician and use it alongside formal testing, symptoms, and medical history rather than as a standalone diagnosis.",
        "Focus on structured follow-up, glucose monitoring, and steady lifestyle improvement while the clinical picture is clarified.",
    ]


def build_recommendations(
    cluster_id: int,
    predicted_class: str | None = None,
    patient_raw: pd.DataFrame | None = None,
    missing_columns: list[str] | None = None,
):
    if cluster_profiles is None or cluster_id not in cluster_profiles.index:
        return dbc.Alert(
            "Cluster profiles are not available yet. Re-run src/train_models.py to generate them.",
            color="warning",
            className="analysis-alert",
        )

    profile = cluster_profiles.loc[cluster_id]
    cluster_name = get_cluster_display_name(cluster_id)
    segment_key = get_cluster_segment_key(cluster_id)
    cluster_meaning = build_cluster_meaning(cluster_id)
    recommendation_lines = []
    meaning_items = build_stage_meaning_items(predicted_class)
    stage_guidance = build_stage_guidance(predicted_class)
    next_steps = build_stage_next_steps(predicted_class)
    combined_meaning_note = build_combined_meaning_note(predicted_class, segment_key)
    patient_specific_actions = build_patient_specific_actions(patient_raw, missing_columns, predicted_class)
    personalisation_note = build_personalisation_note(missing_columns)

    if cluster_meaning:
        meaning_items.append(
            f"The assigned segment is the {cluster_name}, which means the patient most closely matches a group in the training data with a similar overall lifestyle and metabolic pattern."
        )
        meaning_items.append(cluster_meaning["summary"])
    if combined_meaning_note:
        meaning_items.append(combined_meaning_note)

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
                html.H5("Lifestyle Guidance", className="card-title"),
                html.P(
                    f"The predicted stage shows which diabetes pattern the model thinks fits best, and the segment shows the broader patient profile this case most resembles."
                    if predicted_class
                    else f"The patient aligns with the {cluster_name}. These recommendations are based on the assigned patient segment.",
                    className="mb-3",
                ),
                html.Div("What This Means", className="support-label"),
                html.Ul([html.Li(text) for text in meaning_items], className="mb-3"),
                html.Div("Guidance Based on This Stage", className="support-label"),
                html.Ul([html.Li(text) for text in stage_guidance], className="mb-3"),
                html.Div("Priority Actions Based on Current Inputs", className="support-label"),
                html.P(personalisation_note, className="mb-2 text-muted"),
                html.Ul([html.Li(text) for text in patient_specific_actions], className="mb-3"),
                html.Div("Background Pattern From This Segment", className="support-label"),
                html.P(f"Average weekly activity: {profile.get('physical_activity_minutes_per_week', np.nan):.1f} minutes"),
                html.P(f"Average diet score: {profile.get('diet_score', np.nan):.1f}"),
                html.P(f"Average BMI: {profile.get('bmi', np.nan):.1f}"),
                html.Ul([html.Li(text) for text in recommendation_lines], className="mb-3"),
                html.Div("What To Do Next", className="support-label"),
                html.Ul([html.Li(text) for text in next_steps], className="mb-0"),
            ]
        ),
        className="analysis-card",
    )


def build_global_shap_summary_content(model_name: str | None):
    resolved_model_name = normalise_model_name(model_name)
    model_label = get_model_label(resolved_model_name)
    summary_file = ASSETS_DIR / f"{get_model_asset_prefix(resolved_model_name)}_shap_summary.png"

    if summary_file.exists():
        return [
            html.P(
                f"This view shows the overall SHAP distribution across the sampled evaluation records for {model_label}.",
                className="support-copy",
            ),
            html.Img(src=app.get_asset_url(summary_file.name), className="insight-image"),
        ]

    return [
        dbc.Alert(
            f"{model_label} SHAP summary image is missing. Run src/shap_analysis.py --model {resolved_model_name}.",
            color="warning",
            className="analysis-alert mb-0",
        ),
    ]


def build_global_shap_children(model_name: str = DEFAULT_MODEL_NAME):
    default_importance_view = get_default_global_importance_view(model_name)

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Global SHAP Summary", className="card-title"),
                            html.Div(
                                id="global-shap-summary-content",
                                children=build_global_shap_summary_content(model_name),
                            ),
                        ]
                    ),
                    className="analysis-card figure-surface",
                ),
                width=12,
                className="mb-3",
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Global Feature Importance by Stage", className="card-title"),
                            html.P(
                                "Choose a diabetes stage to see which features matter most for that outcome across the evaluation sample.",
                                className="support-copy",
                            ),
                            dcc.Dropdown(
                                id="global-importance-view",
                                options=build_global_importance_options(model_name),
                                value=default_importance_view,
                                placeholder="No class-level SHAP data available",
                                clearable=False,
                                disabled=not build_global_importance_options(model_name),
                                className="wellness-dropdown global-importance-dropdown mb-3",
                            ),
                            dcc.Graph(
                                id="global-importance-figure",
                                figure=build_global_importance_figure(default_importance_view, model_name),
                                config={"displayModeBar": False, "staticPlot": True},
                                responsive=True,
                                className="centered-graph global-importance-graph",
                            ),
                        ]
                    ),
                    className="analysis-card figure-card figure-surface global-importance-card",
                ),
                width=12,
                className="mb-3",
            ),
        ]
    )


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=str(ASSETS_DIR),
)
app.index_string = build_index_string(load_theme_css())
app.title = "Diabetes Risk Decision Support System"
server = app.server

app.layout = build_layout(
    primary_inputs=[build_input_field(column) for column in PRIMARY_FEATURES],
    secondary_inputs=[build_input_field(column) for column in SECONDARY_FEATURES],
    global_shap_children=build_global_shap_children(),
    model_selector=build_model_selector(),
    model_label=html.Span(get_model_label(DEFAULT_MODEL_NAME), id="active-model-label"),
    feature_count=len(FEATURE_COLS),
    training_rows=len(X_train),
    probability_figure=build_empty_figure("Run a prediction to see class probabilities."),
    local_shap_figure=build_empty_figure("Submit a risk prediction to see local SHAP drivers."),
)


@app.callback(
    Output("active-model-label", "children"),
    Output("global-shap-summary-content", "children"),
    Output("global-importance-view", "options"),
    Output("global-importance-view", "value"),
    Input("model-selector", "value"),
)
def update_selected_model(selected_model):
    resolved_model_name = normalise_model_name(selected_model)
    options = build_global_importance_options(resolved_model_name)
    return (
        get_model_label(resolved_model_name),
        build_global_shap_summary_content(resolved_model_name),
        options,
        get_default_global_importance_view(resolved_model_name),
    )


@app.callback(
    Output("global-importance-figure", "figure"),
    Input("model-selector", "value"),
    Input("global-importance-view", "value"),
)
def update_global_importance_figure(selected_model, selected_column):
    return build_global_importance_figure(selected_column, selected_model)


@app.callback(
    Output("prediction-output", "children"),
    Output("probability-figure", "figure"),
    Output("local-shap-summary", "children"),
    Output("local-shap-figure", "figure"),
    Input("predict-btn", "n_clicks"),
    Input("model-selector", "value"),
    [State(f"input-{column}", "value") for column in FEATURE_COLS],
)
def predict_risk(n_clicks, selected_model, *values):
    if not n_clicks:
        return (
            "",
            build_empty_figure("Run a prediction to see class probabilities."),
            "",
            build_empty_figure("Submit a risk prediction to see local SHAP drivers."),
        )

    resolved_model_name = normalise_model_name(selected_model)
    model_label = get_model_label(resolved_model_name)
    raw_inputs = dict(zip(FEATURE_COLS, values))
    patient_raw, patient_encoded, _, _ = prepare_patient_features(raw_inputs)

    model = get_prediction_model(resolved_model_name)
    predicted_index = int(model.predict(patient_encoded)[0])
    predicted_class = target_encoder.inverse_transform([predicted_index])[0]

    proba_frame = pd.Series(model.predict_proba(patient_encoded)[0], index=model.classes_)
    aligned_probabilities = proba_frame.reindex(range(len(target_encoder.classes_)), fill_value=0.0).to_numpy()
    confidence = float(aligned_probabilities[predicted_index])
    ranked_indices = np.argsort(aligned_probabilities)[::-1]
    runner_up_index = int(ranked_indices[1]) if len(ranked_indices) > 1 else predicted_index
    runner_up_class = target_encoder.inverse_transform([runner_up_index])[0]
    runner_up_confidence = float(aligned_probabilities[runner_up_index])
    confidence_gap = max(confidence - runner_up_confidence, 0.0)

    prediction_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4(f"Predicted Diabetes Stage: {predicted_class}", className="card-title"),
                html.P(
                    f"{model_label} found this patient most consistent with the {predicted_class} stage based on the current lifestyle, demographic, and clinical inputs.",
                    className="mb-2",
                ),
                html.P(
                    f"{model_label} confidence is {confidence * 100:.1f}%, meaning {predicted_class} received the highest predicted probability among all available diabetes stages. This reflects model certainty, not a confirmed diagnosis.",
                    className="mb-2",
                ),
                html.P(
                    f"The next most likely stage is {runner_up_class} at {runner_up_confidence * 100:.1f}%, so the gap between the top two stages is {confidence_gap * 100:.1f} percentage points.",
                    className="mb-0 text-muted",
                ),
            ]
        ),
        className="analysis-card analysis-card--accent",
    )

    probability_figure = build_probability_figure(aligned_probabilities, predicted_class)
    _, shap_summary, shap_figure = build_local_shap_outputs(
        patient_raw,
        patient_encoded,
        resolved_model_name,
    )

    return prediction_card, probability_figure, shap_summary, shap_figure


@app.callback(
    Output("cluster-output", "children"),
    Output("recommendations-output", "children"),
    Input("cluster-btn", "n_clicks"),
    Input("predict-btn", "n_clicks"),
    Input("model-selector", "value"),
    [State(f"input-{column}", "value") for column in FEATURE_COLS],
)
def assign_cluster(cluster_clicks, predict_clicks, selected_model, *values):
    if not cluster_clicks and not predict_clicks:
        return dash.no_update, dash.no_update

    resolved_model_name = normalise_model_name(selected_model)
    raw_inputs = dict(zip(FEATURE_COLS, values))
    patient_raw, patient_encoded, patient_scaled, missing_columns = prepare_patient_features(raw_inputs)
    cluster_id = int(kmeans_model.predict(patient_scaled)[0])
    cluster_name = get_cluster_display_name(cluster_id)
    cluster_meaning = build_cluster_meaning(cluster_id)
    predicted_index = int(get_prediction_model(resolved_model_name).predict(patient_encoded)[0])
    predicted_class = target_encoder.inverse_transform([predicted_index])[0]

    triggered_input = None
    if dash.callback_context.triggered:
        triggered_input = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if triggered_input == "cluster-btn":
        cluster_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H4(f"Assigned Segment: {cluster_name}", className="card-title"),
                    html.P(cluster_meaning["summary"] if cluster_meaning else "This segment groups patients with a similar overall profile.", className="mb-3"),
                    html.Div("What This Segment Means", className="support-label"),
                    html.Ul(
                        [html.Li(detail) for detail in (cluster_meaning["details"] if cluster_meaning else [])],
                        className="mb-3",
                    ),
                ]
            ),
            className="analysis-card",
        )
    else:
        cluster_card = dash.no_update

    recommendations = build_recommendations(
        cluster_id,
        predicted_class=predicted_class,
        patient_raw=patient_raw,
        missing_columns=missing_columns,
    )
    return cluster_card, recommendations


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )