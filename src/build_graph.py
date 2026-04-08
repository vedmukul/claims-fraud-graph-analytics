"""
Knowledge Graph Construction
Builds a heterogeneous graph from claims data with 5 node types and 6 relationship types.
Computes graph metrics using NetworkX.
"""
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import logging
from collections import Counter
from src.config import *

logger = logging.getLogger(__name__)


def build_knowledge_graph(claims, patients, providers, pharmacies):
    """
    Construct a knowledge graph with:
      Nodes: Patient, Provider, Pharmacy, Diagnosis, Procedure
      Edges: TREATED_BY, PRESCRIBED_AT, DIAGNOSED_WITH, PERFORMED, REFERRED_TO, BILLED_FOR
    """
    logger.info("Building knowledge graph...")
    G = nx.Graph()

    # ─── Add Nodes ──────────────────────────────────────────────────────────
    for _, row in patients.iterrows():
        G.add_node(row["patient_id"], type="Patient", is_fraud=row["is_fraud"],
                   age=row["age"], gender=row["gender"], state=row["state"],
                   lat=row["lat"], lon=row["lon"], insurance=row["insurance_type"])

    for _, row in providers.iterrows():
        G.add_node(row["provider_id"], type="Provider", is_fraud=row["is_fraud"],
                   specialty=row["specialty"], state=row["state"],
                   lat=row["lat"], lon=row["lon"])

    for _, row in pharmacies.iterrows():
        G.add_node(row["pharmacy_id"], type="Pharmacy",
                   chain=row["chain"], state=row["state"],
                   lat=row["lat"], lon=row["lon"])

    # Add diagnosis and procedure nodes
    diag_codes = claims["diagnosis_code"].unique()
    proc_codes = claims["procedure_code"].unique()
    for d in diag_codes:
        G.add_node(d, type="Diagnosis")
    for p in proc_codes:
        G.add_node(p, type="Procedure")

    logger.info(f"  Nodes: {G.number_of_nodes():,}")

    # ─── Add Edges ──────────────────────────────────────────────────────────
    # Aggregate claims into edge weights
    logger.info("  Adding TREATED_BY edges (patient -> provider)...")
    treated_by = claims.groupby(["patient_id", "provider_id"]).agg(
        weight=("claim_id", "count"),
        total_amount=("claim_amount", "sum"),
        avg_amount=("claim_amount", "mean"),
    ).reset_index()
    for _, row in treated_by.iterrows():
        G.add_edge(row["patient_id"], row["provider_id"],
                   relation="TREATED_BY", weight=row["weight"],
                   total_amount=row["total_amount"], avg_amount=row["avg_amount"])

    logger.info("  Adding PRESCRIBED_AT edges (patient -> pharmacy)...")
    prescribed_at = claims.groupby(["patient_id", "pharmacy_id"]).agg(
        weight=("claim_id", "count"),
    ).reset_index()
    for _, row in prescribed_at.iterrows():
        G.add_edge(row["patient_id"], row["pharmacy_id"],
                   relation="PRESCRIBED_AT", weight=row["weight"])

    logger.info("  Adding DIAGNOSED_WITH edges (patient -> diagnosis)...")
    diagnosed = claims.groupby(["patient_id", "diagnosis_code"]).agg(
        weight=("claim_id", "count"),
    ).reset_index()
    for _, row in diagnosed.iterrows():
        G.add_edge(row["patient_id"], row["diagnosis_code"],
                   relation="DIAGNOSED_WITH", weight=row["weight"])

    logger.info("  Adding PERFORMED edges (provider -> procedure)...")
    performed = claims.groupby(["provider_id", "procedure_code"]).agg(
        weight=("claim_id", "count"),
        avg_charge=("claim_amount", "mean"),
    ).reset_index()
    for _, row in performed.iterrows():
        G.add_edge(row["provider_id"], row["procedure_code"],
                   relation="PERFORMED", weight=row["weight"],
                   avg_charge=row["avg_charge"])

    logger.info("  Adding BILLED_FOR edges (provider -> pharmacy)...")
    billed = claims.groupby(["provider_id", "pharmacy_id"]).agg(
        weight=("claim_id", "count"),
    ).reset_index()
    for _, row in billed.iterrows():
        G.add_edge(row["provider_id"], row["pharmacy_id"],
                   relation="BILLED_FOR", weight=row["weight"])

    # ─── Infer REFERRED_TO edges (sequential patient visits) ────────────────
    logger.info("  Inferring REFERRED_TO edges...")
    claims_sorted = claims.sort_values(["patient_id", "claim_date"])
    referral_counts = Counter()
    for patient_id, group in claims_sorted.groupby("patient_id"):
        providers_seq = group["provider_id"].values
        for i in range(len(providers_seq) - 1):
            if providers_seq[i] != providers_seq[i + 1]:
                pair = (providers_seq[i], providers_seq[i + 1])
                referral_counts[pair] += 1

    for (p1, p2), count in referral_counts.items():
        if count >= 3:  # Only significant referral patterns
            if G.has_edge(p1, p2):
                G[p1][p2]["referral_weight"] = count
            else:
                G.add_edge(p1, p2, relation="REFERRED_TO", weight=count, referral_weight=count)

    logger.info(f"  Edges: {G.number_of_edges():,}")
    logger.info(f"  Graph density: {nx.density(G):.6f}")

    # Save graph
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    logger.info(f"  Graph saved to {GRAPH_PATH}")

    return G


def compute_graph_metrics(G):
    """Compute centrality and community metrics."""
    logger.info("Computing graph metrics...")

    # Extract subgraphs by node type
    provider_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Provider"]
    patient_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Patient"]

    # ─── Centrality Metrics (on full graph) ─────────────────────────────────
    logger.info("  Computing degree centrality...")
    degree_cent = nx.degree_centrality(G)

    logger.info("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, weight="weight")

    logger.info("  Computing betweenness centrality (sampled)...")
    betweenness = nx.betweenness_centrality(G, k=min(1, len(G)), weight="weight")

    # ─── Community Detection (Louvain on provider-patient subgraph) ─────────
    logger.info("  Running Louvain community detection...")
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight", random_state=RANDOM_SEED)
    except ImportError:
        logger.warning("  python-louvain not installed, using greedy modularity")
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G, weight="weight")
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i

    n_communities = len(set(partition.values()))
    logger.info(f"  Found {n_communities} communities")

    metrics = {
        "degree_centrality": degree_cent,
        "pagerank": pagerank,
        "betweenness": betweenness,
        "community": partition,
    }

    # Save metrics
    metrics_path = DATA_DIR / "graph_metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    logger.info(f"  Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    claims = pd.read_parquet(DATA_DIR / "claims.parquet")
    patients = pd.read_parquet(DATA_DIR / "patients.parquet")
    providers = pd.read_parquet(DATA_DIR / "providers.parquet")
    pharmacies = pd.read_parquet(DATA_DIR / "pharmacies.parquet")
    G = build_knowledge_graph(claims, patients, providers, pharmacies)
    metrics = compute_graph_metrics(G)
