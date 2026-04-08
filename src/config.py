"""Central configuration for the fraud detection pipeline."""
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Data generation parameters
N_PATIENTS = 50_000
N_PROVIDERS = 5_000
N_PHARMACIES = 500
N_CLAIMS = 2_000_000
FRAUD_RATE = 0.05  # 5% of entities involved in fraud
RANDOM_SEED = 42

# Graph parameters
GRAPH_PATH = DATA_DIR / "knowledge_graph.gpickle"

# Model parameters
TEST_SIZE = 0.2
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
MAX_DEPTH = 7
ISOLATION_CONTAMINATION = 0.05
ENSEMBLE_WEIGHT_XGB = 0.7
ENSEMBLE_WEIGHT_ISO = 0.3

# Feature groups
TABULAR_FEATURES = [
    "claim_amount", "claim_amount_zscore", "procedure_complexity",
    "claims_per_week", "avg_claim_amount", "std_claim_amount",
    "unique_diagnosis_count", "unique_procedure_count",
    "weekend_claim_ratio", "night_claim_ratio",
    "days_since_last_claim", "claim_amount_percentile",
    "provider_avg_charge", "provider_claim_volume",
    "patient_age", "patient_claim_count",
]

GRAPH_FEATURES = [
    "provider_degree_centrality", "provider_pagerank",
    "provider_betweenness", "provider_community_id",
    "provider_community_size", "provider_community_avg_charge",
    "provider_community_charge_deviation",
    "provider_referral_concentration", "provider_unique_patients",
    "provider_unique_pharmacies",
    "patient_provider_diversity", "patient_pharmacy_shopping_score",
    "patient_geographic_dispersion", "patient_temporal_velocity",
    "patient_unique_diagnoses_30d", "patient_controlled_substance_ratio",
    "patient_community_overlap",
    "pharmacy_unique_providers", "pharmacy_controlled_ratio",
    "pharmacy_patient_volume", "pharmacy_geographic_spread",
    "edge_weight_patient_provider", "edge_weight_provider_pharmacy",
    "referral_pair_frequency", "billing_pair_anomaly_score",
]
