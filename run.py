"""
Pipeline Orchestrator
Runs the full fraud detection pipeline end-to-end:
  1. Generate synthetic data
  2. Build knowledge graph
  3. Compute graph metrics
  4. Engineer features (tabular + graph)
  5. Train ensemble model
"""
import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("=" * 70)
    logger.info("  INSURANCE CLAIMS FRAUD DETECTION PIPELINE")
    logger.info("  Graph Analytics + Ensemble ML + Explainable AI")
    logger.info("=" * 70)

    import pandas as pd
    import pickle
    from src.config import DATA_DIR
    
    if (DATA_DIR / "claims.parquet").exists() and (DATA_DIR / "graph_metrics.pkl").exists():
        logger.info("\n[1-3/5] LOADING EXISTING DATA AND GRAPH METRICS...")
        claims = pd.read_parquet(DATA_DIR / "claims.parquet")
        patients = pd.read_parquet(DATA_DIR / "patients.parquet")
        providers = pd.read_parquet(DATA_DIR / "providers.parquet")
        pharmacies = pd.read_parquet(DATA_DIR / "pharmacies.parquet")
        with open(DATA_DIR / "graph_metrics.pkl", "rb") as f:
            graph_metrics = pickle.load(f)
    else:
        # Step 1: Generate synthetic data
        logger.info("\n[1/5] FETCHING REAL CMS DATA...")
        from src.fetch_real_cms_data import generate_all
        claims, patients, providers, pharmacies = generate_all()

        # Step 2: Build knowledge graph
        logger.info("\n[2/5] BUILDING KNOWLEDGE GRAPH...")
        from src.build_graph import build_knowledge_graph, compute_graph_metrics
        G = build_knowledge_graph(claims, patients, providers, pharmacies)

        # Step 3: Compute graph metrics
        logger.info("\n[3/5] COMPUTING GRAPH METRICS...")
        graph_metrics = compute_graph_metrics(G)

    # Step 4: Feature engineering
    logger.info("\n[4/5] ENGINEERING FEATURES...")
    from src.feature_engineering import build_feature_matrix
    feature_matrix = build_feature_matrix(claims, patients, providers, pharmacies, graph_metrics)

    # Step 5: Train model
    logger.info("\n[5/5] TRAINING ENSEMBLE MODEL...")
    from src.train_model import train_pipeline
    results = train_pipeline()

    elapsed = time.time() - start
    logger.info("\n" + "=" * 70)
    logger.info(f"  PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"  Ensemble AUC-ROC: {results['ensemble_auc']:.4f}")
    logger.info(f"  AUC improvement from graph features: +{results['auc_improvement']*100:.1f} points")
    logger.info(f"  Precision at top 1%: {results['precision_at_top1_pct']:.1%}")
    logger.info("=" * 70)
    logger.info("\n  Launch dashboard: streamlit run dashboards/app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
