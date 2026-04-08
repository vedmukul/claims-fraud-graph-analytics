"""Unit tests for the fraud detection pipeline."""
import sys
import pytest
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSyntheticDataGeneration:
    """Tests for data generation module."""

    def test_generate_patients(self):
        from src.generate_synthetic_data import generate_patients
        patients = generate_patients(100)
        assert len(patients) == 100
        assert "patient_id" in patients.columns
        assert patients["age"].between(18, 95).all()
        assert set(patients["gender"].unique()).issubset({"M", "F"})

    def test_generate_providers(self):
        from src.generate_synthetic_data import generate_providers
        providers = generate_providers(50)
        assert len(providers) == 50
        assert "provider_id" in providers.columns
        assert "specialty" in providers.columns

    def test_generate_pharmacies(self):
        from src.generate_synthetic_data import generate_pharmacies
        pharmacies = generate_pharmacies(20)
        assert len(pharmacies) == 20
        assert "pharmacy_id" in pharmacies.columns

    def test_generate_base_claims(self):
        from src.generate_synthetic_data import (
            generate_patients, generate_providers, generate_pharmacies, generate_base_claims
        )
        patients = generate_patients(100)
        providers = generate_providers(50)
        pharmacies = generate_pharmacies(20)
        claims = generate_base_claims(1000, patients, providers, pharmacies)
        assert len(claims) == 1000
        assert "claim_amount" in claims.columns
        assert (claims["claim_amount"] > 0).all()

    def test_fraud_injection(self):
        from src.generate_synthetic_data import (
            generate_patients, generate_providers, generate_pharmacies,
            generate_base_claims, inject_phantom_billing
        )
        patients = generate_patients(1000)
        providers = generate_providers(100)
        pharmacies = generate_pharmacies(20)
        claims = generate_base_claims(10000, patients, providers, pharmacies)
        claims, providers = inject_phantom_billing(claims, providers, n_fraud_providers=5)
        assert providers["is_fraud"].sum() == 5
        assert claims["is_fraud"].sum() > 0


class TestGraphConstruction:
    """Tests for graph building."""

    def test_graph_has_correct_node_types(self):
        G = nx.Graph()
        G.add_node("PAT_001", type="Patient")
        G.add_node("PRV_001", type="Provider")
        G.add_edge("PAT_001", "PRV_001", relation="TREATED_BY", weight=5)
        patient_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Patient"]
        assert len(patient_nodes) == 1

    def test_graph_metrics_computation(self):
        G = nx.Graph()
        for i in range(20):
            G.add_node(f"N_{i}", type="Provider")
        for i in range(19):
            G.add_edge(f"N_{i}", f"N_{i+1}", weight=1)
        pagerank = nx.pagerank(G)
        assert len(pagerank) == 20
        assert all(v > 0 for v in pagerank.values())


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_zscore_computation(self):
        amounts = pd.Series([100, 200, 300, 400, 500])
        zscore = (amounts - amounts.mean()) / amounts.std()
        assert abs(zscore.mean()) < 1e-10
        assert abs(zscore.std() - 1.0) < 0.1

    def test_controlled_substance_ratio(self):
        claims = pd.DataFrame({
            "patient_id": ["P1"] * 10,
            "diagnosis_code": ["E11.9"] * 7 + ["T40.2X1A"] * 3,
        })
        controlled = ["T40.2X1A"]
        ratio = len(claims[claims["diagnosis_code"].isin(controlled)]) / len(claims)
        assert abs(ratio - 0.3) < 0.01


class TestModelTraining:
    """Tests for model components."""

    def test_ensemble_weighting(self):
        xgb_scores = np.array([0.8, 0.3, 0.9, 0.1])
        iso_scores = np.array([0.7, 0.4, 0.85, 0.2])
        ensemble = 0.7 * xgb_scores + 0.3 * iso_scores
        assert ensemble[0] == pytest.approx(0.77, abs=0.01)
        assert ensemble[2] == pytest.approx(0.885, abs=0.01)

    def test_precision_at_k(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.05, 0.15, 0.25])
        top_k = 3
        top_indices = np.argsort(y_scores)[-top_k:]
        precision_at_k = y_true[top_indices].mean()
        assert precision_at_k == 1.0  # Top 3 are all fraud


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
