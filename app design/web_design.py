from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html


THEME_CSS_PATH = Path(__file__).with_name("web_design.css")


def load_theme_css() -> str:
    return THEME_CSS_PATH.read_text(encoding="utf-8")


def build_index_string(css_text: str) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
    <head>
        {{%metas%}}
        <title>Diabetes Risk Decision Support System</title>
        {{%favicon%}}
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
        <link href=\"https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap\" rel=\"stylesheet\">
        {{%css%}}
        <style>
{css_text}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""


def build_empty_state(title: str, description: str, kicker: str | None = None):
    children = []
    if kicker:
        children.append(html.Div(kicker, className="state-kicker"))
    children.extend(
        [
            html.H4(title, className="state-title"),
            html.P(description, className="state-copy"),
        ]
    )
    return html.Div(children, className="empty-state")


def build_input_card(label: str, helper_text: str, control):
    return dbc.Col(
        html.Div(
            [
                html.Label(label, className="input-label"),
                control,
                html.Small(helper_text, className="input-helper"),
            ],
            className="input-card",
        ),
        md=6,
        lg=4,
        className="mb-3",
    )


def build_stat_card(title: str, value: str, note: str):
    return html.Div(
        [
            html.Div(title, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.P(note, className="metric-note"),
        ],
        className="metric-card",
    )


def build_section_header(kicker: str, title: str, description: str):
    return html.Div(
        [
            html.Div(kicker, className="section-kicker"),
            html.H2(title, className="section-title"),
            html.P(description, className="section-copy"),
        ],
        className="section-header",
    )


def build_layout(
    *,
    primary_inputs,
    secondary_inputs,
    global_shap_children,
    model_selector,
    model_label: str,
    feature_count: int,
    training_rows: int,
    probability_figure,
    local_shap_figure,
    cluster_size_children,
    cluster_scatter_children,
):
    return html.Div(
        [
            dbc.Container(
                [
                    html.Div(
                        [
                            build_stat_card("Active model", model_label, "Used for stage prediction and local SHAP"),
                            build_stat_card("Training rows", f"{training_rows:,}", "Current training sample used for defaults"),
                            build_stat_card("Patient inputs", str(feature_count), "Lifestyle, demographic, and clinical features"),
                            build_stat_card("Explainability", "SHAP", "Global plots and patient-level contribution view"),
                        ],
                        className="metric-grid",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div("Model Switcher", className="panel-kicker"),
                                        html.H3("Choose the active stage model", className="panel-title"),
                                        html.P(
                                            "Switch between XGBoost, Decision Tree, and Random Forest. This changes stage prediction, class probabilities, and SHAP explanations. Patient clustering stays the same.",
                                            className="panel-copy",
                                        ),
                                        model_selector,
                                    ],
                                    className="action-panel",
                                ),
                                lg=12,
                                xl=12,
                                className="mx-auto mb-4",
                            ),
                        ],
                        className="g-4 align-items-start",
                    ),
                    html.Section(
                        dcc.Tabs(
                            parent_className="wellness-tabs-parent",
                            className="wellness-tabs",
                            children=[
                                dcc.Tab(
                                    label="Risk Prediction",
                                    className="wellness-tab",
                                    selected_className="wellness-tab wellness-tab--selected",
                                    children=[
                                        html.Div(
                                            [
                                                build_section_header(
                                                    "Patient Intake",
                                                    "Assess diabetes stage risk",
                                                    "Start with the highest-impact fields and expand the optional inputs when you want a more detailed patient profile.",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Row(primary_inputs, className="gx-3 input-grid"),
                                                                dbc.Accordion(
                                                                    [
                                                                        dbc.AccordionItem(
                                                                            [
                                                                                html.P(
                                                                                    "Use these additional fields when you want more detail. Leaving them blank keeps the interface fast while still running the model.",
                                                                                    className="support-copy",
                                                                                ),
                                                                                dbc.Row(secondary_inputs, className="gx-3 input-grid"),
                                                                            ],
                                                                            title="Optional advanced inputs",
                                                                        )
                                                                    ],
                                                                    className="detail-accordion",
                                                                    start_collapsed=True,
                                                                ),
                                                            ],
                                                            lg=12,
                                                            className="mb-4",
                                                        ),
                                                    ],
                                                    className="g-4 align-items-start",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Div("Assessment Action", className="panel-kicker"),
                                                                        html.H3("Generate a patient prediction", className="panel-title"),
                                                                        html.P(
                                                                            "The dashboard will score the patient, show class probabilities, and prepare a local SHAP explanation for the same submission.",
                                                                            className="panel-copy",
                                                                        ),
                                                                        html.Button("Predict Risk", id="predict-btn", className="action-button action-button--primary"),
                                                                    ],
                                                                    className="action-panel",
                                                                ),
                                                                html.Div(
                                                                    id="prediction-output",
                                                                    children=build_empty_state(
                                                                        "Prediction results will appear here",
                                                                        "Submit the patient profile to see the diabetes-stage prediction, confidence score, and model explanation.",
                                                                        kicker="Risk Summary",
                                                                    ),
                                                                    className="surface-card surface-card--stacked",
                                                                ),
                                                            ],
                                                            lg=12,
                                                            xl=12,
                                                            className="mx-auto mb-4",
                                                        ),
                                                    ],
                                                    className="g-4 align-items-start",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Div("Probability Profile", className="panel-kicker"),
                                                                    html.H3("Model confidence by diabetes stage", className="panel-title"),
                                                                    dcc.Graph(
                                                                        id="probability-figure",
                                                                        figure=probability_figure,
                                                                        config={"displayModeBar": False, "staticPlot": True},
                                                                        responsive=True,
                                                                        className="centered-graph",
                                                                    ),
                                                                ],
                                                                className="surface-card figure-card probability-card",
                                                            ),
                                                            lg=12,
                                                            xl=12,
                                                            className="mx-auto",
                                                        ),
                                                    ],
                                                    className="g-4 align-items-start",
                                                ),
                                            ],
                                            className="tab-section",
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Patient Segmentation",
                                    className="wellness-tab",
                                    selected_className="wellness-tab wellness-tab--selected",
                                    children=[
                                        html.Div(
                                            [
                                                build_section_header(
                                                    "Cluster Assignment",
                                                    "Place the patient into a peer segment",
                                                    "Segmentation uses the same prepared patient record as the prediction view, so your cluster output stays aligned with the risk profile.",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Div("Segmentation Action", className="panel-kicker"),
                                                                        html.H3("Assign lifestyle cluster", className="panel-title"),
                                                                        html.P(
                                                                            "Use the patient details already entered in the prediction tab. This will update both the cluster summary and the recommendations tab.",
                                                                            className="panel-copy",
                                                                        ),
                                                                        html.Button("Assign Cluster", id="cluster-btn", className="action-button action-button--secondary"),
                                                                    ],
                                                                    className="action-panel",
                                                                ),
                                                                html.Div(
                                                                    id="cluster-output",
                                                                    children=build_empty_state(
                                                                        "Cluster output will appear here",
                                                                        "Run the segmentation step after entering patient details to see the assigned patient segment and what it means.",
                                                                        kicker="Peer Segment",
                                                                    ),
                                                                    className="surface-card surface-card--stacked",
                                                                ),
                                                            ],
                                                            lg=12,
                                                            xl=12,
                                                            className="mx-auto mb-4",
                                                        ),
                                                    ],
                                                    className="g-4 align-items-start",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Div("Cluster Sizes", className="panel-kicker"),
                                                                    html.H3("Training patients per segment", className="panel-title"),
                                                                    html.P(
                                                                        "This chart uses the saved training cluster assignments and shows how large each K-means segment is in the training data.",
                                                                        className="panel-copy",
                                                                    ),
                                                                    html.Div(
                                                                        cluster_size_children,
                                                                        id="cluster-size-content",
                                                                    ),
                                                                ],
                                                                className="surface-card figure-card cluster-size-card",
                                                            ),
                                                            lg=12,
                                                            xl=12,
                                                            className="mx-auto mb-4",
                                                        ),
                                                    ],
                                                    className="g-4 align-items-stretch",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Div("Cluster Map", className="panel-kicker"),
                                                                    html.H3("2D view of patient segments", className="panel-title"),
                                                                    html.P(
                                                                        "The scatter plot uses PCA on the scaled training data so the three K-means segments can be compared visually in two dimensions.",
                                                                        className="panel-copy",
                                                                    ),
                                                                    html.Div(
                                                                        cluster_scatter_children,
                                                                        id="cluster-scatter-content",
                                                                    ),
                                                                ],
                                                                className="surface-card figure-card cluster-scatter-card",
                                                            ),
                                                            lg=12,
                                                            xl=12,
                                                            className="mx-auto mb-4",
                                                        ),
                                                    ],
                                                    className="g-4 align-items-stretch",
                                                ),
                                            ],
                                            className="tab-section",
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="SHAP Insights",
                                    className="wellness-tab",
                                    selected_className="wellness-tab wellness-tab--selected",
                                    children=[
                                        html.Div(
                                            [
                                                build_section_header(
                                                    "Model Explanation",
                                                    "Review global and patient-level drivers",
                                                    "Global SHAP assets describe the overall model. The patient-level chart updates after each risk prediction so you can see what moved the result.",
                                                ),
                                                html.Div(global_shap_children, id="global-shap-output", className="surface-stack mb-4"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="local-shap-summary",
                                                                    children=build_empty_state(
                                                                        "Local SHAP summary will appear here",
                                                                        "Run a risk prediction first to see the most important patient-specific contributors.",
                                                                        kicker="Patient Drivers",
                                                                    ),
                                                                    className="surface-card",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Div("Contribution Chart", className="panel-kicker"),
                                                                        html.H3("How the patient profile moved the model", className="panel-title"),
                                                                        dcc.Graph(
                                                                            id="local-shap-figure",
                                                                            figure=local_shap_figure,
                                                                            config={"displayModeBar": False, "staticPlot": True},
                                                                            responsive=True,
                                                                        ),
                                                                    ],
                                                                    className="surface-card figure-card surface-card--stacked",
                                                                ),
                                                            ],
                                                            lg=12,
                                                            xl=12,
                                                            className="mx-auto mb-4",
                                                        ),
                                                    ],
                                                    className="g-4 align-items-start",
                                                ),
                                            ],
                                            className="tab-section",
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Recommendations",
                                    className="wellness-tab",
                                    selected_className="wellness-tab wellness-tab--selected",
                                    children=[
                                        html.Div(
                                            [
                                                build_section_header(
                                                    "Lifestyle Guidance",
                                                    "Turn patient results into practical recommendations",
                                                    "This panel updates after a risk prediction or patient-cluster assignment. It keeps the final guidance separate from the model mechanics while showing what to do next.",
                                                ),
                                                html.Div(
                                                    id="recommendations-output",
                                                    children=build_empty_state(
                                                        "Recommendations will appear here",
                                                        "Run a risk prediction or assign a patient cluster to generate lifestyle guidance and next-step advice.",
                                                        kicker="Care Guidance",
                                                    ),
                                                    className="surface-card surface-card--tall",
                                                ),
                                            ],
                                            className="tab-section",
                                        )
                                    ],
                                ),
                            ],
                        ),
                        className="tabs-shell",
                    ),
                ],
                fluid=True,
                className="page-frame",
            )
        ],
        className="app-shell",
    )