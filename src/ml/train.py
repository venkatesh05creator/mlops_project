import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.ml.data_loader import load_analytical_data
from src.ml.features import (
    SEVERITY_INT_TO_LABEL,
    SEVERITY_LABEL_TO_INT,
    TrafficFeatureExtractor,
    create_chronological_splits,
    get_preprocessor,
)
from src.utils.logger import get_logger

logger = get_logger("ml_train")


def plot_and_save_confusion_matrix(cm, class_names, output_path: Path, title: str):
    """Generates and saves a publication-quality confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Ground Truth Severity",
        xlabel="Predicted Severity",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_and_save_feature_importances(
    feature_names, importances, output_path: Path, top_n: int = 15
):
    """Plots and saves top N feature importances."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = [importances[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores[::-1], align="center", color="#1E88E5")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Relative Importance Score")
    ax.set_title(f"Top {top_n} Engineered Features")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_predictions(
    y_true, y_pred, split_name: str = "val"
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """
    Computes comprehensive multi-class metrics plus pooled high-severity recall and false-negative rate.
    """
    labels = [0, 1, 2]
    class_names = [SEVERITY_INT_TO_LABEL[i] for i in labels]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    per_class_rec = np.asarray(
        recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    )
    per_class_f1 = np.asarray(
        f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    )

    # Pooled High-Severity (Serious [1] + Fatal [2]) metrics
    y_true_high = (y_true >= 1).astype(int)
    y_pred_high = (y_pred >= 1).astype(int)
    high_sev_recall = recall_score(y_true_high, y_pred_high, zero_division=0)
    high_sev_fnr = 1.0 - high_sev_recall  # Missed severe accidents rate

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = cast(
        dict[str, Any],
        classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        ),
    )

    metrics = {
        f"{split_name}_accuracy": float(acc),
        f"{split_name}_macro_f1": float(macro_f1),
        f"{split_name}_weighted_f1": float(weighted_f1),
        f"{split_name}_macro_recall": float(macro_recall),
        f"{split_name}_macro_precision": float(macro_prec),
        f"{split_name}_recall_slight": float(per_class_rec[0]),
        f"{split_name}_recall_serious": float(per_class_rec[1]),
        f"{split_name}_recall_fatal": float(per_class_rec[2]),
        f"{split_name}_f1_slight": float(per_class_f1[0]),
        f"{split_name}_f1_serious": float(per_class_f1[1]),
        f"{split_name}_f1_fatal": float(per_class_f1[2]),
        f"{split_name}_high_severity_recall": float(high_sev_recall),
        f"{split_name}_high_severity_fnr": float(high_sev_fnr),
    }

    return metrics, cm, report


def train_and_track_models():
    """
    Executes Phase 2 & Phase 3 end-to-end:
    - Loads data and performs chronological splitting
    - Applies feature extraction & preprocessor
    - Calculates class weights for imbalance mitigation
    - Trains Random Forest, XGBoost, and LightGBM
    - Logs runs, parameters, metrics, and visual artifacts to MLflow
    - Compares validation performance and registers the best model
    """
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {config.MLFLOW_EXPERIMENT_NAME}")

    # 1. Load Data & Chronological Split
    df = load_analytical_data()
    train_df, val_df, test_df = create_chronological_splits(df)

    y_train = train_df["accident_severity_label"].map(SEVERITY_LABEL_TO_INT).values
    y_val = val_df["accident_severity_label"].map(SEVERITY_LABEL_TO_INT).values
    y_test = test_df["accident_severity_label"].map(SEVERITY_LABEL_TO_INT).values

    # Feature transformation
    extractor = TrafficFeatureExtractor()
    preprocessor, num_cols, cat_cols = get_preprocessor()

    X_train_extracted = extractor.fit_transform(train_df)
    X_val_extracted = extractor.transform(val_df)
    X_test_extracted = extractor.transform(test_df)

    X_train_proc = preprocessor.fit_transform(X_train_extracted)
    X_val_proc = preprocessor.transform(X_val_extracted)
    X_test_proc = preprocessor.transform(X_test_extracted)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_ohe_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(cat_ohe_names)

    # Calculate Balanced Class Weights
    classes = [0, 1, 2]
    class_weights = compute_class_weight(
        "balanced", classes=np.array(classes), y=y_train
    )
    class_weight_dict = {i: float(w) for i, w in zip(classes, class_weights)}
    class_weight_dict_str = {
        SEVERITY_INT_TO_LABEL[i]: float(w) for i, w in zip(classes, class_weights)
    }
    train_sample_weights = compute_sample_weight("balanced", y_train)

    logger.info(f"Computed Class Weights: {class_weight_dict_str}")

    artifacts_scratch_dir = BASE_DIR / "models" / "artifacts"
    artifacts_scratch_dir.mkdir(parents=True, exist_ok=True)

    # Model Configurations
    models_to_train = {
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 150,
                "max_depth": 12,
                "min_samples_split": 5,
                "imbalance_strategy": "balanced_class_weight",
            },
            "fit_kwargs": {},
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            ),
            "params": {
                "model_type": "LGBMClassifier",
                "n_estimators": 150,
                "max_depth": 8,
                "learning_rate": 0.05,
                "imbalance_strategy": "balanced_class_weight",
            },
            "fit_kwargs": {},
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "model_type": "XGBClassifier",
                "n_estimators": 150,
                "max_depth": 6,
                "learning_rate": 0.05,
                "imbalance_strategy": "balanced_sample_weights",
            },
            "fit_kwargs": {"sample_weight": train_sample_weights},
        },
    }

    run_results = {}
    best_model_name = None
    best_val_score = -1.0
    best_pipeline = None

    class_names = [SEVERITY_INT_TO_LABEL[i] for i in classes]

    for model_name, config_dict in models_to_train.items():
        logger.info(
            f"\n==================== TRAINING {model_name} ===================="
        )
        model = config_dict["model"]
        params = config_dict["params"]
        fit_kwargs = config_dict["fit_kwargs"]

        with mlflow.start_run(run_name=f"{model_name}_Severity_Run") as run:
            run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_dict(class_weight_dict_str, "class_weights.json")

            # Fit model on preprocessed training data
            logger.info(f"Fitting {model_name} on training set...")
            model.fit(X_train_proc, y_train, **fit_kwargs)

            # Validation Predictions & Evaluation
            val_preds = model.predict(X_val_proc)
            val_metrics, val_cm, val_report = evaluate_predictions(
                y_val, val_preds, split_name="val"
            )

            # Test Predictions & Evaluation
            test_preds = model.predict(X_test_proc)
            test_metrics, test_cm, test_report = evaluate_predictions(
                y_test, test_preds, split_name="test"
            )

            # Log metrics
            mlflow.log_metrics({**val_metrics, **test_metrics})

            logger.info(
                f"{model_name} Results:\n"
                f"  Val Macro-F1: {val_metrics['val_macro_f1']:.4f} | "
                f"  Val Macro-Recall: {val_metrics['val_macro_recall']:.4f} | "
                f"  Val High-Severity Recall: {val_metrics['val_high_severity_recall']:.4f} | "
                f"  Val High-Severity FNR: {val_metrics['val_high_severity_fnr']:.4f}\n"
                f"  Test Macro-F1: {test_metrics['test_macro_f1']:.4f} | "
                f"  Test High-Severity Recall: {test_metrics['test_high_severity_recall']:.4f}"
            )

            # Generate and log Confusion Matrix Artifacts
            cm_val_path = artifacts_scratch_dir / f"cm_val_{model_name}.png"
            plot_and_save_confusion_matrix(
                val_cm,
                class_names,
                cm_val_path,
                f"{model_name} - Validation Confusion Matrix",
            )
            mlflow.log_artifact(str(cm_val_path), artifact_path="evaluation_plots")

            cm_test_path = artifacts_scratch_dir / f"cm_test_{model_name}.png"
            plot_and_save_confusion_matrix(
                test_cm,
                class_names,
                cm_test_path,
                f"{model_name} - Test Confusion Matrix",
            )
            mlflow.log_artifact(str(cm_test_path), artifact_path="evaluation_plots")

            # Feature Importances Plot
            if hasattr(model, "feature_importances_"):
                fi_path = artifacts_scratch_dir / f"feat_imp_{model_name}.png"
                plot_and_save_feature_importances(
                    all_feature_names, model.feature_importances_, fi_path
                )
                mlflow.log_artifact(str(fi_path), artifact_path="evaluation_plots")

            # Log full classification reports as JSON
            mlflow.log_dict(val_report, f"classification_report_val_{model_name}.json")
            mlflow.log_dict(
                test_report, f"classification_report_test_{model_name}.json"
            )

            # Package into an end-to-end Pipeline: Raw Features -> TrafficFeatureExtractor -> Preprocessor -> Model
            full_inference_pipeline = Pipeline(
                [
                    ("extractor", extractor),
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            # Save pipeline locally
            pipeline_path = config.MODELS_DIR / f"{model_name.lower()}_pipeline.joblib"
            joblib.dump(full_inference_pipeline, pipeline_path)
            mlflow.log_artifact(str(pipeline_path), artifact_path="saved_joblib")

            # Log standard MLflow Model artifact with cloudpickle
            mlflow.sklearn.log_model(
                sk_model=full_inference_pipeline,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            run_results[model_name] = {
                "run_id": run_id,
                "val_macro_f1": val_metrics["val_macro_f1"],
                "val_high_severity_recall": val_metrics["val_high_severity_recall"],
                "test_macro_f1": test_metrics["test_macro_f1"],
                "test_high_severity_recall": test_metrics["test_high_severity_recall"],
                "pipeline_path": str(pipeline_path),
            }

            # Decision criteria: High-severity recall is critical for safety, balanced with macro-F1
            composite_score = (
                0.5 * val_metrics["val_macro_f1"]
                + 0.5 * val_metrics["val_high_severity_recall"]
            )
            if composite_score > best_val_score:
                best_val_score = composite_score
                best_model_name = model_name
                best_pipeline = full_inference_pipeline

    logger.info("\n==================== COMPARISON & SELECTION ====================")
    for m, res in run_results.items():
        logger.info(
            f"Model {m:12s} | Val Macro-F1: {res['val_macro_f1']:.4f} | "
            f"Val Severe Recall: {res['val_high_severity_recall']:.4f} | "
            f"Test Macro-F1: {res['test_macro_f1']:.4f}"
        )

    logger.info(
        f"\n>>> Selected Best Model: {best_model_name} (Composite Score: {best_val_score:.4f})"
    )

    # Save Best Model as production candidate
    best_prod_pipeline_path = config.MODELS_DIR / "production_pipeline.joblib"
    joblib.dump(best_pipeline, best_prod_pipeline_path)
    logger.info(f"Saved production pipeline to: {best_prod_pipeline_path}")

    # Save metadata info
    best_info = {
        "model_name": best_model_name,
        "run_id": run_results[best_model_name]["run_id"],
        "class_mapping": SEVERITY_LABEL_TO_INT,
        "inverse_class_mapping": SEVERITY_INT_TO_LABEL,
        "metrics": run_results[best_model_name],
    }
    with open(config.MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(best_info, f, indent=2)

    # Register in MLflow Model Registry
    try:
        best_run_id = run_results[best_model_name]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        reg_model = mlflow.register_model(
            model_uri=model_uri, name=config.MODEL_REGISTRY_NAME
        )
        logger.info(
            f"Successfully registered model '{config.MODEL_REGISTRY_NAME}' version {reg_model.version} in MLflow Registry!"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"MLflow model registration notice (local tracking mode): {e}")

    return run_results, best_model_name


if __name__ == "__main__":
    train_and_track_models()
