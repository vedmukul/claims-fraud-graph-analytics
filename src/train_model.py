"""
Ensemble Fraud Detection Model
Combines XGBoost (supervised) + Isolation Forest (unsupervised) with SHAP explainability.
Trains on both tabular and graph features, then evaluates tabular-only vs full model.
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_recall_curve, average_precision_score,
                              classification_report, confusion_matrix, precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import shap
from src.config import *

logger = logging.getLogger(__name__)


def prepare_data(feature_matrix):
    """Split features and target, handle class imbalance."""
    logger.info("Preparing data...")

    y = feature_matrix["is_fraud"].astype(int)
    X = feature_matrix.drop(columns=["is_fraud"])

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)

    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Samples: {X.shape[0]:,}")
    logger.info(f"  Fraud rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost with class weight handling."""
    logger.info("Training XGBoost...")

    # Calculate scale_pos_weight for imbalanced classes
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)

    # Precision at top 1%
    top_1_pct = int(len(y_test) * 0.01)
    top_indices = np.argsort(y_pred_proba)[-top_1_pct:]
    precision_at_top1 = y_test.iloc[top_indices].mean()

    logger.info(f"  XGBoost AUC-ROC: {auc:.4f}")
    logger.info(f"  XGBoost Avg Precision: {ap:.4f}")
    logger.info(f"  Precision at top 1%: {precision_at_top1:.4f}")

    return model, y_pred_proba, auc


def train_isolation_forest(X_train, X_test, graph_feature_cols):
    """Train Isolation Forest on graph features for unsupervised anomaly detection."""
    logger.info("Training Isolation Forest on graph features...")

    # Use only graph features that exist in the data
    available_graph_cols = [c for c in graph_feature_cols if c in X_train.columns]
    if not available_graph_cols:
        logger.warning("  No graph features found for Isolation Forest!")
        return None, np.zeros(len(X_test))

    scaler = StandardScaler()
    X_train_graph = scaler.fit_transform(X_train[available_graph_cols].fillna(0))
    X_test_graph = scaler.transform(X_test[available_graph_cols].fillna(0))

    iso_model = IsolationForest(
        contamination=ISOLATION_CONTAMINATION,
        random_state=RANDOM_SEED,
        n_estimators=200,
        n_jobs=-1,
    )
    iso_model.fit(X_train_graph)

    # Score: lower = more anomalous, convert to 0-1 range
    iso_scores = -iso_model.score_samples(X_test_graph)
    iso_scores_normalized = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)

    logger.info(f"  Isolation Forest trained on {len(available_graph_cols)} graph features")
    return iso_model, iso_scores_normalized


def train_tabular_only_baseline(X_train, y_train, X_test, y_test, graph_feature_cols):
    """Train XGBoost with only tabular features for comparison."""
    logger.info("Training tabular-only baseline...")

    available_graph_cols = [c for c in graph_feature_cols if c in X_train.columns]
    tabular_cols = [c for c in X_train.columns if c not in available_graph_cols]

    if not tabular_cols:
        return 0.0

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    baseline_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="auc",
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=-1,
    )
    baseline_model.fit(X_train[tabular_cols], y_train, verbose=0)
    y_pred_baseline = baseline_model.predict_proba(X_test[tabular_cols])[:, 1]
    baseline_auc = roc_auc_score(y_test, y_pred_baseline)

    logger.info(f"  Tabular-only AUC-ROC: {baseline_auc:.4f}")
    return baseline_auc


def compute_shap_values(model, X_test, feature_names):
    """Compute SHAP values for model explainability."""
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:5000])  # Sample for speed
    logger.info("  SHAP values computed")
    return explainer, shap_values


def ensemble_predictions(xgb_proba, iso_scores):
    """Combine XGBoost and Isolation Forest scores."""
    ensemble = (ENSEMBLE_WEIGHT_XGB * xgb_proba + ENSEMBLE_WEIGHT_ISO * iso_scores)
    return ensemble


def train_pipeline():
    """Full training pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)

    # Load feature matrix
    feature_matrix = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(feature_matrix)

    # Define graph feature columns (those present in data)
    graph_feature_cols = [c for c in GRAPH_FEATURES if c in X_train.columns]

    # ─── Train tabular-only baseline ────────────────────────────────────────
    baseline_auc = train_tabular_only_baseline(X_train, y_train, X_test, y_test, graph_feature_cols)

    # ─── Train full XGBoost (tabular + graph) ───────────────────────────────
    xgb_model, xgb_proba, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test, feature_names)

    # ─── Train Isolation Forest ─────────────────────────────────────────────
    iso_model, iso_scores = train_isolation_forest(X_train, X_test, graph_feature_cols)

    # ─── Ensemble ───────────────────────────────────────────────────────────
    ensemble_scores = ensemble_predictions(xgb_proba, iso_scores)
    ensemble_auc = roc_auc_score(y_test, ensemble_scores)

    # Precision at top 1%
    top_1_pct = int(len(y_test) * 0.01)
    top_indices = np.argsort(ensemble_scores)[-top_1_pct:]
    precision_at_top1 = y_test.iloc[top_indices].mean()

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Tabular-only AUC-ROC:   {baseline_auc:.4f}")
    logger.info(f"  Full XGBoost AUC-ROC:   {xgb_auc:.4f}")
    logger.info(f"  Ensemble AUC-ROC:       {ensemble_auc:.4f}")
    logger.info(f"  AUC improvement:        +{(ensemble_auc - baseline_auc):.4f} ({(ensemble_auc - baseline_auc)*100:.1f} points)")
    logger.info(f"  Precision at top 1%:    {precision_at_top1:.4f}")
    logger.info("=" * 60)

    # ─── SHAP Explainability ────────────────────────────────────────────────
    explainer, shap_values = compute_shap_values(xgb_model, X_test, feature_names)

    # ─── Feature Importance ─────────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature": [c for c in X_train.columns],
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("\nTop 15 Features:")
    for _, row in importance.head(15).iterrows():
        source = "GRAPH" if row["feature"] in graph_feature_cols else "TABULAR"
        logger.info(f"  [{source}] {row['feature']}: {row['importance']:.4f}")

    # ─── Save artifacts ─────────────────────────────────────────────────────
    pickle.dump(xgb_model, open(MODEL_DIR / "xgboost_model.pkl", "wb"))
    pickle.dump(iso_model, open(MODEL_DIR / "isolation_forest.pkl", "wb"))
    pickle.dump(explainer, open(MODEL_DIR / "shap_explainer.pkl", "wb"))
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    # Save test data and predictions for dashboard
    test_data = X_test.copy()
    test_data["is_fraud"] = y_test.values
    test_data["fraud_score_xgb"] = xgb_proba
    test_data["fraud_score_iso"] = iso_scores
    test_data["fraud_score_ensemble"] = ensemble_scores
    test_data.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)

    # Save SHAP values for dashboard
    np.save(MODEL_DIR / "shap_values.npy", shap_values[:5000])
    X_test[:5000].to_parquet(DATA_DIR / "shap_sample.parquet", index=False)

    # Save results summary
    results = {
        "baseline_auc": round(baseline_auc, 4),
        "xgboost_auc": round(xgb_auc, 4),
        "ensemble_auc": round(ensemble_auc, 4),
        "auc_improvement": round(ensemble_auc - baseline_auc, 4),
        "precision_at_top1_pct": round(precision_at_top1, 4),
        "n_features_total": len(feature_names),
        "n_graph_features": len(graph_feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    json.dump(results, open(MODEL_DIR / "results.json", "w"), indent=2)

    logger.info(f"\nAll artifacts saved to {MODEL_DIR}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_pipeline()
