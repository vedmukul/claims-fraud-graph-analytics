"""
Feature Engineering: Tabular + Graph Features
Extracts 50+ features combining traditional claims analytics with graph-derived signals.
"""
import pandas as pd
import numpy as np
import pickle
import logging
from scipy.spatial.distance import cdist
from src.config import *

logger = logging.getLogger(__name__)


def compute_tabular_features(claims, patients, providers):
    """Compute traditional tabular features from claims data."""
    logger.info("Computing tabular features...")

    # ─── Claim-level features ───────────────────────────────────────────────
    claims["claim_amount_zscore"] = (
        (claims["claim_amount"] - claims["claim_amount"].mean()) / claims["claim_amount"].std()
    )
    claims["claim_amount_percentile"] = claims["claim_amount"].rank(pct=True)

    # ─── Provider-level aggregates ──────────────────────────────────────────
    provider_stats = claims.groupby("provider_id").agg(
        provider_avg_charge=("claim_amount", "mean"),
        provider_std_charge=("claim_amount", "std"),
        provider_claim_volume=("claim_id", "count"),
        provider_unique_patients_tab=("patient_id", "nunique"),
        provider_unique_procedures=("procedure_code", "nunique"),
        provider_unique_diagnoses=("diagnosis_code", "nunique"),
        provider_weekend_ratio=("is_weekend", "mean"),
        provider_night_ratio=("is_night", "mean"),
        provider_avg_complexity=("procedure_complexity", "mean"),
    ).reset_index()
    provider_stats["provider_std_charge"] = provider_stats["provider_std_charge"].fillna(0)

    # ─── Burst Billing Time-Series Anomaly ──────────────────────────────────
    claims["claim_month"] = claims["claim_date"].dt.month
    monthly_claims = claims.groupby(["provider_id", "claim_month"]).size().unstack(fill_value=0)
    
    provider_burst_stats = pd.DataFrame(index=monthly_claims.index)
    provider_burst_stats["provider_max_month_claims"] = monthly_claims.max(axis=1)
    provider_burst_stats["provider_median_month_claims"] = monthly_claims.median(axis=1).clip(lower=1) # Avoid div 0
    provider_burst_stats["provider_burst_month_ratio"] = provider_burst_stats["provider_max_month_claims"] / provider_burst_stats["provider_median_month_claims"]
    
    provider_stats = provider_stats.merge(provider_burst_stats.reset_index(), on="provider_id", how="left")

    # ─── Patient-level aggregates ───────────────────────────────────────────
    patient_stats = claims.groupby("patient_id").agg(
        patient_claim_count=("claim_id", "count"),
        patient_avg_amount=("claim_amount", "mean"),
        patient_std_amount=("claim_amount", "std"),
        patient_unique_providers=("provider_id", "nunique"),
        patient_unique_pharmacies=("pharmacy_id", "nunique"),
        patient_unique_diagnoses_total=("diagnosis_code", "nunique"),
        patient_unique_procedures=("procedure_code", "nunique"),
        claims_per_week=("claim_id", "count"),  # will adjust below
        weekend_claim_ratio=("is_weekend", "mean"),
        night_claim_ratio=("is_night", "mean"),
    ).reset_index()
    patient_stats["patient_std_amount"] = patient_stats["patient_std_amount"].fillna(0)

    # Claims per week (normalize by date range)
    date_range_weeks = max(
        (claims["claim_date"].max() - claims["claim_date"].min()).days / 7, 1
    )
    patient_stats["claims_per_week"] = patient_stats["patient_claim_count"] / date_range_weeks

    # Days since last claim per patient
    last_claim = claims.groupby("patient_id")["claim_date"].max().reset_index()
    last_claim.columns = ["patient_id", "last_claim_date"]
    max_date = claims["claim_date"].max()
    last_claim["days_since_last_claim"] = (max_date - last_claim["last_claim_date"]).dt.days
    patient_stats = patient_stats.merge(last_claim[["patient_id", "days_since_last_claim"]], on="patient_id", how="left")

    # ─── Controlled substance ratios ────────────────────────────────────────
    controlled_codes = ["T40.2X1A", "F11.20", "F10.20", "Z79.891"]
    controlled_claims = claims[claims["diagnosis_code"].isin(controlled_codes)]
    controlled_ratio = controlled_claims.groupby("patient_id").size() / claims.groupby("patient_id").size()
    controlled_ratio = controlled_ratio.reset_index()
    controlled_ratio.columns = ["patient_id", "controlled_substance_ratio"]
    patient_stats = patient_stats.merge(controlled_ratio, on="patient_id", how="left")
    patient_stats["controlled_substance_ratio"] = patient_stats["controlled_substance_ratio"].fillna(0)

    # ─── Merge patient age and insurance ────────────────────────────────────
    patient_stats = patient_stats.merge(
        patients[["patient_id", "age", "insurance_type"]],
        on="patient_id", how="left"
    )
    patient_stats.rename(columns={"age": "patient_age"}, inplace=True)

    return claims, provider_stats, patient_stats


def compute_graph_features(claims, patients, providers, pharmacies, graph_metrics):
    """Compute graph-derived features from the knowledge graph metrics."""
    logger.info("Computing graph features...")

    degree_cent = graph_metrics["degree_centrality"]
    pagerank = graph_metrics["pagerank"]
    betweenness = graph_metrics["betweenness"]
    community = graph_metrics["community"]

    # ─── Provider graph features ────────────────────────────────────────────
    provider_graph = []
    for pid in providers["provider_id"]:
        provider_graph.append({
            "provider_id": pid,
            "provider_degree_centrality": degree_cent.get(pid, 0),
            "provider_pagerank": pagerank.get(pid, 0),
            "provider_betweenness": betweenness.get(pid, 0),
            "provider_community_id": community.get(pid, -1),
        })
    provider_graph_df = pd.DataFrame(provider_graph)

    # Community-level aggregates
    community_charges = claims.groupby("provider_id")["claim_amount"].mean().reset_index()
    community_charges.columns = ["provider_id", "avg_charge"]
    community_charges = community_charges.merge(
        provider_graph_df[["provider_id", "provider_community_id"]], on="provider_id"
    )

    community_stats = community_charges.groupby("provider_community_id").agg(
        community_avg_charge=("avg_charge", "mean"),
        community_std_charge=("avg_charge", "std"),
        provider_community_size=("provider_id", "count"),
    ).reset_index()
    community_stats["community_std_charge"] = community_stats["community_std_charge"].fillna(0)

    provider_graph_df = provider_graph_df.merge(community_stats, on="provider_community_id", how="left")
    provider_graph_df["provider_community_avg_charge"] = provider_graph_df["community_avg_charge"]

    # Deviation from community average
    provider_charges = community_charges[["provider_id", "avg_charge"]]
    provider_graph_df = provider_graph_df.merge(provider_charges, on="provider_id", how="left")
    provider_graph_df["provider_community_charge_deviation"] = (
        (provider_graph_df["avg_charge"] - provider_graph_df["community_avg_charge"]) /
        provider_graph_df["community_std_charge"].replace(0, 1)
    )

    # Referral concentration
    ref_pairs = claims.sort_values(["patient_id", "claim_date"])
    ref_pairs["next_provider"] = ref_pairs.groupby("patient_id")["provider_id"].shift(-1)
    
    valid_refs = ref_pairs[
        ref_pairs["next_provider"].notna() &
        (ref_pairs["provider_id"] != ref_pairs["next_provider"])
    ]
    
    ref_counts = valid_refs.groupby(["provider_id", "next_provider"]).size().reset_index(name="count")
    
    if not ref_counts.empty:
        ref_counts = ref_counts.sort_values(["provider_id", "count"], ascending=[True, False])
        top3_sum = ref_counts.groupby("provider_id").head(3).groupby("provider_id")["count"].sum()
        total_sum = ref_counts.groupby("provider_id")["count"].sum()
        ref_concentration = (top3_sum / total_sum).reset_index(name="provider_referral_concentration")
    else:
        ref_concentration = pd.DataFrame({"provider_id": [], "provider_referral_concentration": []})
        
    provider_graph_df = provider_graph_df.merge(ref_concentration, on="provider_id", how="left")
    provider_graph_df["provider_referral_concentration"] = provider_graph_df["provider_referral_concentration"].fillna(0)

    # Unique patients and pharmacies per provider (graph-based)
    provider_graph_df["provider_unique_patients"] = claims.groupby("provider_id")["patient_id"].nunique().reindex(provider_graph_df["provider_id"]).values
    provider_graph_df["provider_unique_pharmacies"] = claims.groupby("provider_id")["pharmacy_id"].nunique().reindex(provider_graph_df["provider_id"]).values

    # ─── Patient graph features ─────────────────────────────────────────────
    pat_grouped = claims.groupby("patient_id")
    n_providers = pat_grouped["provider_id"].nunique()
    n_pharmacies = pat_grouped["pharmacy_id"].nunique()
    claim_counts = pat_grouped.size()
    
    pharmacy_shopping = (n_pharmacies / claim_counts.clip(lower=1)) * 10
    
    dates_max = pat_grouped["claim_date"].max()
    dates_min = pat_grouped["claim_date"].min()
    date_ranges = (dates_max - dates_min).dt.days
    temporal_velocity = np.where(claim_counts > 1, claim_counts / date_ranges.div(7).clip(lower=1), 0)
    
    # Fast vectorized geographic dispersion computation
    claims_geo = claims[["patient_id", "provider_id"]].copy()
    claims_geo = claims_geo.merge(patients[["patient_id", "lat", "lon"]].rename(columns={"lat":"pat_lat", "lon":"pat_lon"}), on="patient_id", how="left")
    claims_geo = claims_geo.merge(providers[["provider_id", "lat", "lon"]].rename(columns={"lat":"prov_lat", "lon":"prov_lon"}), on="provider_id", how="left")
    
    # Approx Euclidean metric
    claims_geo["distance_to_provider"] = np.sqrt(
        (claims_geo["pat_lat"].astype(float) - claims_geo["prov_lat"].astype(float))**2 + 
        (claims_geo["pat_lon"].astype(float) - claims_geo["prov_lon"].astype(float))**2
    )
    pat_dispersion = claims_geo.groupby("patient_id")["distance_to_provider"].mean().fillna(0)
    prov_dispersion = claims_geo.groupby("provider_id")["distance_to_provider"].mean().fillna(0)
    
    # Inject Provider dispersion feature directly into provider_graph_df
    provider_graph_df = provider_graph_df.set_index("provider_id")
    provider_graph_df["provider_geographic_dispersion"] = prov_dispersion
    provider_graph_df = provider_graph_df.reset_index()
    
    # Fast community overlap
    claims_comm = claims[["patient_id", "provider_id"]].copy()
    claims_comm["comm"] = claims_comm["provider_id"].map(community).fillna(-1)
    pat_comm_overlap = claims_comm.groupby("patient_id")["comm"].nunique().fillna(0)

    patient_graph_df = pd.DataFrame({
        "patient_provider_diversity": n_providers,
        "patient_pharmacy_shopping_score": pharmacy_shopping.round(3),
        "patient_geographic_dispersion": pat_dispersion.round(3),
        "patient_temporal_velocity": np.round(temporal_velocity, 3),
        "patient_community_overlap": pat_comm_overlap,
    }).reset_index()
    patient_graph_df = patients[["patient_id"]].merge(patient_graph_df, on="patient_id", how="left").fillna(0)

    # ─── Pharmacy graph features ────────────────────────────────────────────
    controlled_codes = ["T40.2X1A", "F11.20", "F10.20", "Z79.891"]
    pharmacy_graph = claims.groupby("pharmacy_id").agg(
        pharmacy_unique_providers=("provider_id", "nunique"),
        pharmacy_patient_volume=("patient_id", "nunique"),
    ).reset_index()

    controlled_pharm = claims[claims["diagnosis_code"].isin(controlled_codes)].groupby("pharmacy_id").size()
    total_pharm = claims.groupby("pharmacy_id").size()
    pharm_controlled_ratio = (controlled_pharm / total_pharm).reset_index()
    pharm_controlled_ratio.columns = ["pharmacy_id", "pharmacy_controlled_ratio"]
    pharmacy_graph = pharmacy_graph.merge(pharm_controlled_ratio, on="pharmacy_id", how="left")
    pharmacy_graph["pharmacy_controlled_ratio"] = pharmacy_graph["pharmacy_controlled_ratio"].fillna(0)

    # Geographic spread of a pharmacy's patients
    pharm_spread = []
    for phid in pharmacies["pharmacy_id"]:
        pat_ids = claims[claims["pharmacy_id"] == phid]["patient_id"].unique()
        pat_locs = patients[patients["patient_id"].isin(pat_ids)][["lat", "lon"]].values
        if len(pat_locs) > 1:
            pharm_spread.append({"pharmacy_id": phid, "pharmacy_geographic_spread": pat_locs.std()})
        else:
            pharm_spread.append({"pharmacy_id": phid, "pharmacy_geographic_spread": 0})
    pharm_spread_df = pd.DataFrame(pharm_spread)
    pharmacy_graph = pharmacy_graph.merge(pharm_spread_df, on="pharmacy_id", how="left")

    return provider_graph_df, patient_graph_df, pharmacy_graph


def build_feature_matrix(claims, patients, providers, pharmacies, graph_metrics):
    """Build the complete feature matrix for model training."""
    logger.info("=" * 60)
    logger.info("BUILDING FEATURE MATRIX")
    logger.info("=" * 60)

    # Compute features
    claims, provider_stats, patient_stats = compute_tabular_features(claims, patients, providers)
    provider_graph_df, patient_graph_df, pharmacy_graph = compute_graph_features(
        claims, patients, providers, pharmacies, graph_metrics
    )

    # ─── Build claim-level feature matrix ───────────────────────────────────
    logger.info("Merging features into claim-level matrix (Optimized for Memory)...")
    
    # Cast to category to save 90% of memory on ID strings
    for col in ["patient_id", "provider_id", "pharmacy_id"]:
        claims[col] = claims[col].astype("category")
        if col in provider_stats.columns: provider_stats[col] = provider_stats[col].astype("category")
        if col in provider_graph_df.columns: provider_graph_df[col] = provider_graph_df[col].astype("category")
        if col in patient_stats.columns: patient_stats[col] = patient_stats[col].astype("category")
        if col in patient_graph_df.columns: patient_graph_df[col] = patient_graph_df[col].astype("category")
        if col in pharmacy_graph.columns: pharmacy_graph[col] = pharmacy_graph[col].astype("category")

    features = claims[["claim_id", "patient_id", "provider_id", "pharmacy_id",
                        "claim_amount", "claim_amount_zscore", "procedure_complexity",
                        "claim_amount_percentile", "is_weekend", "is_night", "is_fraud"]].copy()

    # Merge provider tabular
    features = features.merge(provider_stats, on="provider_id", how="left")
    import gc
    del provider_stats; gc.collect()
    
    # Merge provider graph
    cols_to_use = [c for c in provider_graph_df.columns if c != "avg_charge"]
    features = features.merge(cols_to_use and provider_graph_df[cols_to_use], on="provider_id", how="left")
    del provider_graph_df; gc.collect()
    
    # Merge patient tabular
    features = features.merge(patient_stats, on="patient_id", how="left")
    del patient_stats; gc.collect()
    
    # Merge patient graph
    features = features.merge(patient_graph_df, on="patient_id", how="left")
    del patient_graph_df; gc.collect()
    
    # Merge pharmacy graph
    features = features.merge(pharmacy_graph, on="pharmacy_id", how="left")
    del pharmacy_graph; gc.collect()

    # ─── Edge-weight features ───────────────────────────────────────────────
    edge_patient_provider = claims.groupby(["patient_id", "provider_id"], observed=True).size().reset_index(name="edge_weight_patient_provider")
    features = features.merge(edge_patient_provider, on=["patient_id", "provider_id"], how="left")
    del edge_patient_provider; gc.collect()

    edge_provider_pharmacy = claims.groupby(["provider_id", "pharmacy_id"], observed=True).size().reset_index(name="edge_weight_provider_pharmacy")
    features = features.merge(edge_provider_pharmacy, on=["provider_id", "pharmacy_id"], how="left")
    del edge_provider_pharmacy; gc.collect()

    # Referral pair frequency
    ref_pairs = claims.sort_values(["patient_id", "claim_date"])
    ref_pairs["next_provider"] = ref_pairs.groupby("patient_id", observed=True)["provider_id"].shift(-1)
    features = features.merge(ref_pairs[["claim_id", "next_provider"]], on="claim_id", how="left")
    
    ref_pair_freq = ref_pairs.groupby(["provider_id", "next_provider"], observed=True).size().reset_index(name="referral_pair_frequency")
    features = features.merge(ref_pair_freq, on=["provider_id", "next_provider"], how="left")
    features["referral_pair_frequency"] = features["referral_pair_frequency"].fillna(0)

    # Billing pair anomaly score
    features["billing_pair_anomaly_score"] = (
        features["provider_community_charge_deviation"].abs() *
        features["provider_pagerank"]
    ).fillna(0)

    # Fill remaining NAs
    # Ensure 0 is a valid category before fillna
    for col in features.select_dtypes(include=["category"]).columns:
        if 0 not in features[col].cat.categories:
            features[col] = features[col].cat.add_categories([0])
            
    features = features.fillna(0)
    logger.info(f"Feature matrix complete. Shape: {features.shape}")

    # Encode categoricals
    if "insurance_type" in features.columns:
        features = pd.get_dummies(features, columns=["insurance_type"], drop_first=True)

    # Drop ID columns and non-feature columns
    drop_cols = ["claim_id", "patient_id", "provider_id", "pharmacy_id",
                 "next_provider", "community_avg_charge", "community_std_charge"]
    drop_cols = [c for c in drop_cols if c in features.columns]
    feature_matrix = features.drop(columns=drop_cols)

    # Save
    feature_matrix.to_parquet(DATA_DIR / "feature_matrix.parquet", index=False)
    logger.info(f"Feature matrix: {feature_matrix.shape[0]:,} rows x {feature_matrix.shape[1]} columns")
    logger.info(f"Fraud rate: {feature_matrix['is_fraud'].mean():.2%}")
    logger.info(f"Saved to {DATA_DIR / 'feature_matrix.parquet'}")

    return feature_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    claims = pd.read_parquet(DATA_DIR / "claims.parquet")
    patients = pd.read_parquet(DATA_DIR / "patients.parquet")
    providers = pd.read_parquet(DATA_DIR / "providers.parquet")
    pharmacies = pd.read_parquet(DATA_DIR / "pharmacies.parquet")
    with open(DATA_DIR / "graph_metrics.pkl", "rb") as f:
        graph_metrics = pickle.load(f)
    build_feature_matrix(claims, patients, providers, pharmacies, graph_metrics)
