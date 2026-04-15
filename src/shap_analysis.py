"""Generate SHAP outputs for the trained diabetes risk models.

Pipeline prerequisite:
1. src/prepare_data.py
2. src/preprocess_data.py
3. src/train_models2.py

By default this script analyses the saved Random Forest model so the output
filenames match the current Dash app's SHAP tab. Pass --model xgboost if you
want SHAP outputs for the best-performing classifier from train_models2.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

MODEL_FILES = {
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl",
}

MODEL_LABELS = {
    "decision_tree": "DecisionTree",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis on a trained diabetes risk model.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FILES),
        default="random_forest",
        help="Trained model to explain.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=500,
        help="Number of X_train rows to use as SHAP background data.",
    )
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=1000,
        help="Number of X_test rows to explain for the global plots.",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=15,
        help="Maximum number of features to display in the plots.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling background and evaluation rows.",
    )
    parser.add_argument(
        "--waterfall-index",
        type=int,
        default=0,
        help="Row index inside the sampled evaluation frame for the local plot.",
    )
    parser.add_argument(
        "--skip-waterfall",
        action="store_true",
        help="Skip the local SHAP waterfall plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory for the SHAP PNG outputs.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for the SHAP feature-importance CSV.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}. Run prepare_data.py, "
            "preprocess_data.py, and train_models2.py first."
        )
    return path


def load_pickle(path: Path):
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def sample_frame(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(frame):
        return frame.copy()
    return frame.sample(n=sample_size, random_state=random_state).copy()


def load_pipeline_outputs(model_name: str):
    model = load_pickle(require_file(MODELS_DIR / MODEL_FILES[model_name]))
    target_encoder = load_pickle(require_file(ARTIFACTS_DIR / "target_encoder.pkl"))
    X_train = pd.read_csv(require_file(DATA_DIR / "X_train.csv"))
    X_test = pd.read_csv(require_file(DATA_DIR / "X_test.csv"))
    return model, target_encoder, X_train, X_test


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


def predicted_class_matrix(values: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values

    projected = np.empty((values.shape[0], values.shape[1]), dtype=float)
    for row_index, class_index in enumerate(predictions.astype(int)):
        projected[row_index] = values[row_index, :, class_index]
    return projected


def build_feature_importance(
    values: np.ndarray,
    projected_values: np.ndarray,
    feature_names: list[str],
    class_names: np.ndarray,
) -> pd.DataFrame:
    importance_data = {
        "feature": feature_names,
        "mean_abs_shap_predicted_class": np.abs(projected_values).mean(axis=0),
    }

    if values.ndim == 3:
        importance_data["mean_abs_shap_all_classes"] = np.abs(values).mean(axis=(0, 2))
        for class_index, class_name in enumerate(class_names):
            column_name = f"mean_abs_shap_{class_name}"
            importance_data[column_name] = np.abs(values[:, :, class_index]).mean(axis=0)
    else:
        importance_data["mean_abs_shap_all_classes"] = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame(importance_data)
    return importance_df.sort_values(
        "mean_abs_shap_predicted_class",
        ascending=False,
    )


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


def save_summary_plot(
    projected_values: np.ndarray,
    X_eval: pd.DataFrame,
    model_label: str,
    output_path: Path,
    max_display: int,
) -> None:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        projected_values,
        X_eval,
        plot_type="dot",
        max_display=max_display,
        show=False,
    )
    plt.title(f"{model_label} SHAP Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_bar_plot(
    importance_df: pd.DataFrame,
    model_label: str,
    output_path: Path,
    max_display: int,
) -> None:
    top_features = importance_df.head(max_display).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature"],
        top_features["mean_abs_shap_predicted_class"],
        color="#2f6db3",
    )
    plt.xlabel("Mean |SHAP| for predicted class")
    plt.title(f"{model_label} SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_waterfall_plot(
    values: np.ndarray,
    base_values: np.ndarray,
    X_eval: pd.DataFrame,
    predictions: np.ndarray,
    class_names: np.ndarray,
    model_label: str,
    sample_index: int,
    output_path: Path,
    max_display: int,
) -> None:
    bounded_index = max(0, min(sample_index, len(X_eval) - 1))

    if values.ndim == 3:
        class_index = int(predictions[bounded_index])
        sample_values = values[bounded_index, :, class_index]
        base_value = resolve_base_value(base_values, bounded_index, class_index)
        class_label = class_names[class_index]
    else:
        class_index = int(predictions[bounded_index]) if len(class_names) > 1 else 0
        sample_values = values[bounded_index]
        base_value = resolve_base_value(base_values, bounded_index, class_index)
        class_label = class_names[class_index] if len(class_names) > class_index else "prediction"

    explanation = shap.Explanation(
        values=sample_values,
        base_values=base_value,
        data=X_eval.iloc[bounded_index].to_numpy(),
        feature_names=X_eval.columns.tolist(),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.title(
        f"{model_label} SHAP Waterfall - row {bounded_index} ({class_label})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    csv_dir = args.csv_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    model, target_encoder, X_train, X_test = load_pipeline_outputs(args.model)

    background = sample_frame(X_train, args.background_size, args.random_state)
    X_eval = sample_frame(X_test, args.evaluation_size, args.random_state)

    explainer = shap.TreeExplainer(model, data=background)
    values, base_values = compute_shap_values(explainer, X_eval)

    predictions = np.asarray(model.predict(X_eval)).astype(int)
    projected_values = predicted_class_matrix(values, predictions)
    class_names = np.asarray(target_encoder.classes_)

    model_label = MODEL_LABELS[args.model]
    summary_path = output_dir / f"{model_label}_shap_summary.png"
    bar_path = output_dir / f"{model_label}_shap_bar.png"
    waterfall_path = output_dir / f"{model_label}_shap_waterfall.png"
    csv_path = csv_dir / f"{args.model}_shap_feature_importance.csv"

    importance_df = build_feature_importance(
        values,
        projected_values,
        X_eval.columns.tolist(),
        class_names,
    )
    importance_df.to_csv(csv_path, index=False)

    save_summary_plot(
        projected_values,
        X_eval,
        model_label,
        summary_path,
        args.max_display,
    )
    save_bar_plot(
        importance_df,
        model_label,
        bar_path,
        args.max_display,
    )

    if not args.skip_waterfall:
        save_waterfall_plot(
            values,
            base_values,
            X_eval,
            predictions,
            class_names,
            model_label,
            args.waterfall_index,
            waterfall_path,
            args.max_display,
        )

    print(f"Model analysed: {args.model}")
    print(f"Background rows: {len(background)}")
    print(f"Evaluation rows: {len(X_eval)}")
    print(f"Saved summary plot: {summary_path}")
    print(f"Saved bar plot: {bar_path}")
    if not args.skip_waterfall:
        print(f"Saved waterfall plot: {waterfall_path}")
    print(f"Saved feature importance CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())