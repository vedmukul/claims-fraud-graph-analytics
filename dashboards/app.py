"""
Fraud Detection Dashboard
Interactive investigation interface with SHAP explanations, graph visualizations,
and ranked fraud flags.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from pathlib import Path
import networkx as nx
from pyvis.network import Network
import tempfile

# ─── Config ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Claims Fraud Detection", page_icon="\U0001F50D", layout="wide")


@st.cache_data
def load_data():
    test_preds = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    results = json.load(open(MODEL_DIR / "results.json"))
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    shap_values = np.load(MODEL_DIR / "shap_values.npy", allow_pickle=True)
    shap_sample = pd.read_parquet(DATA_DIR / "shap_sample.parquet")
    claims = pd.read_parquet(DATA_DIR / "claims.parquet")
    providers = pd.read_parquet(DATA_DIR / "providers.parquet")
    patients = pd.read_parquet(DATA_DIR / "patients.parquet")
    return test_preds, results, importance, shap_values, shap_sample, claims, providers, patients


test_preds, results, importance, shap_values, shap_sample, claims, providers, patients = load_data()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🕵️‍♀️ Fraud Detection")
st.sidebar.info("👋 **Welcome!** This tool helps you catch Medicare fraud by analyzing complex billing behaviors using Artificial Intelligence.")
page = st.sidebar.radio("Navigate", [
    "📊 1. Overall AI Performance",
    "🚨 2. Suspicious Doctors Queue",
    "🧠 3. Why was it flagged? (AI Logic)",
    "🕸️ 4. Doctor & Patient Network",
    "🏘️ 5. Fraud Ring Detection",
    "🛡️ 6. Check My Doctor (For Patients)",
])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 1. Overall AI Performance":
    st.title("📊 Overall AI Performance")
    st.info("💡 **What is this?** This page shows how well our Artificial Intelligence model is at catching fraud. An **AUC-ROC** score of 1.0 means the AI is 100% perfectly accurate at separating legitimate doctors from fraudsters.")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ensemble AUC-ROC", f"{results['ensemble_auc']:.3f}")
    col2.metric("vs. Tabular Baseline", f"+{results['auc_improvement']*100:.1f} pts",
                delta=f"{results['auc_improvement']*100:.1f}")
    col3.metric("Precision @ Top 1%", f"{results['precision_at_top1_pct']:.1%}")
    col4.metric("Graph Features", f"{results['n_graph_features']}")

    st.divider()

    # AUC Comparison
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("AUC-ROC Comparison")
        auc_df = pd.DataFrame({
            "Model": ["Tabular Only", "XGBoost (Full)", "Ensemble"],
            "AUC-ROC": [results["baseline_auc"], results["xgboost_auc"], results["ensemble_auc"]],
        })
        fig = px.bar(auc_df, x="Model", y="AUC-ROC", color="Model",
                     color_discrete_sequence=["#ccc", "#4A90D9", "#0F3A5F"],
                     text_auto=".3f")
        fig.update_layout(showlegend=False, yaxis_range=[0.5, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top 15 Feature Importance")
        top15 = importance.head(15).copy()
        graph_features = [c for c in top15["feature"] if "provider_" in c or "patient_" in c
                          or "pharmacy_" in c or "community" in c or "edge_" in c
                          or "referral" in c or "billing" in c]
        top15["source"] = top15["feature"].apply(lambda x: "Graph" if x in graph_features else "Tabular")
        fig = px.bar(top15, x="importance", y="feature", color="source", orientation="h",
                     color_discrete_map={"Graph": "#0F3A5F", "Tabular": "#A0C4E8"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Score Distribution
    st.subheader("Fraud Score Distribution")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Legitimate Claims", "Fraudulent Claims"))
    legit = test_preds[test_preds["is_fraud"] == 0]["fraud_score_ensemble"]
    fraud = test_preds[test_preds["is_fraud"] == 1]["fraud_score_ensemble"]
    fig.add_trace(go.Histogram(x=legit, nbinsx=50, marker_color="#A0C4E8", name="Legitimate"), row=1, col=1)
    fig.add_trace(go.Histogram(x=fraud, nbinsx=50, marker_color="#D32F2F", name="Fraud"), row=1, col=2)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: FLAGGED CASES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 2. Suspicious Doctors Queue":
    st.title("🚨 Suspicious Doctors Queue")
    st.info("💡 **What is this?** Here is the prioritized list of the highest-risk claims/doctors. Our AI has ranked them from most suspicious to least suspicious. Investigators should start reviewing the cases at the top of this list.")

    # Top flagged cases
    n_flags = st.slider("Number of flags to show", 10, 200, 50)
    top_flagged = test_preds.nlargest(n_flags, "fraud_score_ensemble")

    # Summary
    true_fraud = top_flagged["is_fraud"].sum()
    st.metric(f"True Fraud in Top {n_flags}", f"{true_fraud}/{n_flags} ({true_fraud/n_flags:.0%} precision)")

    # Display table
    display_cols = ["fraud_score_ensemble", "is_fraud", "claim_amount", "claim_amount_zscore",
                    "provider_pagerank", "patient_pharmacy_shopping_score",
                    "provider_community_charge_deviation", "patient_temporal_velocity"]
    available_cols = [c for c in display_cols if c in top_flagged.columns]
    st.dataframe(
        top_flagged[available_cols].style.background_gradient(subset=["fraud_score_ensemble"], cmap="Reds"),
        use_container_width=True, height=500
    )

    # Risk factor breakdown
    st.subheader("Risk Factor Distribution in Flagged Cases")
    if "provider_community_charge_deviation" in top_flagged.columns:
        fig = px.scatter(top_flagged, x="provider_pagerank", y="provider_community_charge_deviation",
                         color="is_fraud", size="claim_amount",
                         color_discrete_map={True: "#D32F2F", False: "#A0C4E8"},
                         labels={"provider_pagerank": "Provider PageRank",
                                 "provider_community_charge_deviation": "Community Billing Deviation"},
                         title="PageRank vs Community Billing Deviation")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 3. Why was it flagged? (AI Logic)":
    st.title("🧠 Why was it flagged? (AI Logic)")
    st.info("💡 **What is this?** Modern AI can be a 'black box'. This page opens the box to show you *exactly* which factors made a specific doctor look suspicious (e.g., they charge way more than their peers, or they share too many patients with a known fraudster).")

    st.subheader("Global Feature Impact (Beeswarm Plot)")
    import matplotlib.pyplot as plt
    fig_summary, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample, max_display=20, show=False)
    st.pyplot(fig_summary, use_container_width=True)
    plt.close()

    st.divider()

    st.subheader("Individual Case Explanation")
    case_idx = st.number_input("Select case index (0-4999)", 0, min(4999, len(shap_sample)-1), 0)

    # Waterfall plot
    fig_waterfall, ax2 = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[case_idx],
        base_values=shap_values[case_idx].sum() - shap_values[case_idx].sum(),
        data=shap_sample.iloc[case_idx],
        feature_names=shap_sample.columns.tolist()
    ), max_display=15, show=False)
    st.pyplot(fig_waterfall, use_container_width=True)
    plt.close()

    # Case details
    with st.expander("View raw feature values for this case"):
        case_data = shap_sample.iloc[case_idx]
        st.json(case_data.to_dict())


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: NETWORK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🕸️ 4. Doctor & Patient Network":
    st.title("🕸️ Doctor & Patient Network")
    st.info("💡 **What is this?** Fraudsters rarely work alone. This interactive web shows how doctors, patients, and pharmacies are connected. \n\n**Red circles** = Fraudulent Doctors. **Large circles** = High billing volume.")

    # Build a subgraph of top flagged providers
    n_providers_show = st.slider("Number of top-risk providers to visualize", 5, 30, 10)

    fraud_providers = providers[providers["is_fraud"] == 1].head(n_providers_show)["provider_id"].tolist()
    if not fraud_providers:
        fraud_providers = providers.head(n_providers_show)["provider_id"].tolist()

    # Get their connected patients
    connected_claims = claims[claims["provider_id"].isin(fraud_providers)]
    connected_patients = connected_claims["patient_id"].unique()[:50]  # Limit for viz
    connected_pharmacies = connected_claims["pharmacy_id"].unique()[:20]

    # Build Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    # Add provider nodes
    for pid in fraud_providers:
        prov = providers[providers["provider_id"] == pid].iloc[0]
        vol = len(connected_claims[connected_claims["provider_id"] == pid])
        color = "#D32F2F" if prov["is_fraud"] == 1 else "#4A90D9"
        net.add_node(str(pid), label=str(pid)[:8], color=color, size=max(15, min(vol/5, 50)),
                     title=f"{pid}\nSpecialty: {prov['specialty']}\nClaims: {vol}\nFraud: {prov['is_fraud']}")

    # Add patient nodes
    for pat_id in connected_patients:
        pat = patients[patients["patient_id"] == pat_id]
        if len(pat) > 0:
            pat = pat.iloc[0]
            color = "#FF8A80" if pat["is_fraud"] == 1 else "#81C784"
            net.add_node(str(pat_id), label=str(pat_id)[:12], color=color, size=8,
                         title=f"{pat_id}\nAge: {pat['age']}\nFraud: {pat['is_fraud']}")

    # Add pharmacy nodes
    for phid in connected_pharmacies:
        net.add_node(str(phid), label=str(phid)[:7], color="#FFA726", size=12,
                     title=f"{phid}")

    # Add edges
    for _, claim in connected_claims.iterrows():
        if claim["patient_id"] in connected_patients:
            net.add_edge(str(claim["patient_id"]), str(claim["provider_id"]),
                         color="#ccc", width=1)
        if claim["pharmacy_id"] in connected_pharmacies:
            net.add_edge(str(claim["provider_id"]), str(claim["pharmacy_id"]),
                         color="#E0E0E0", width=0.5)

    # Render
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        with open(f.name, "r") as html_file:
            html_content = html_file.read()
    st.components.v1.html(html_content, height=650, scrolling=True)

    st.caption("""
    **Legend:** \U0001F534 Red = Fraud Provider | \U0001F7E2 Green = Legit Patient |
    \U0001F7E0 Orange = Pharmacy | Pink = Fraud Patient
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: COMMUNITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ 5. Fraud Ring Detection":
    st.title("🏘️ Fraud Ring Detection")
    
    st.info("💡 **What is this?** By looking at which doctors share the same patients, our algorithm groups them into 'Communities' (like cartels or fraud rings). If an entire community has abnormally high charges, it is a massive red flag for organized crime.")

    # Load graph metrics
    try:
        with open(DATA_DIR / "graph_metrics.pkl", "rb") as f:
            graph_metrics = pickle.load(f)

        community = graph_metrics["community"]

        # Build community-level stats
        provider_communities = {pid: community.get(pid, -1) for pid in providers["provider_id"]}
        providers_with_comm = providers.copy()
        providers_with_comm["community_id"] = providers_with_comm["provider_id"].map(provider_communities)

        # Merge with claims for billing stats
        provider_charges = claims.groupby("provider_id").agg(
            avg_charge=("claim_amount", "mean"),
            total_claims=("claim_id", "count"),
            fraud_claims=("is_fraud", "sum"),
        ).reset_index()

        providers_with_comm = providers_with_comm.merge(provider_charges, on="provider_id", how="left")

        comm_stats = providers_with_comm.groupby("community_id").agg(
            n_providers=("provider_id", "count"),
            avg_charge=("avg_charge", "mean"),
            total_claims=("total_claims", "sum"),
            fraud_providers=("is_fraud", "sum"),
            fraud_claims=("fraud_claims", "sum"),
        ).reset_index()
        comm_stats["fraud_rate"] = comm_stats["fraud_providers"] / comm_stats["n_providers"]
        comm_stats["charge_zscore"] = (comm_stats["avg_charge"] - comm_stats["avg_charge"].mean()) / comm_stats["avg_charge"].std()

        # Flag anomalous communities
        comm_stats["is_anomalous"] = comm_stats["charge_zscore"].abs() > 2

        # Display
        st.subheader(f"Community Overview ({len(comm_stats)} communities)")
        n_anomalous = comm_stats["is_anomalous"].sum()
        st.warning(f"\u26A0\uFE0F **{n_anomalous} communities** have billing patterns 2+ standard deviations from the norm")

        # Scatter plot
        fig = px.scatter(comm_stats, x="n_providers", y="avg_charge",
                         size="total_claims", color="is_anomalous",
                         color_discrete_map={True: "#D32F2F", False: "#4A90D9"},
                         hover_data=["community_id", "fraud_rate", "charge_zscore"],
                         labels={"n_providers": "Providers in Community",
                                 "avg_charge": "Avg Charge per Claim",
                                 "is_anomalous": "Anomalous"},
                         title="Community Billing Patterns")
        st.plotly_chart(fig, use_container_width=True)

        # Anomalous communities detail
        st.subheader("Anomalous Communities")
        anomalous = comm_stats[comm_stats["is_anomalous"]].sort_values("charge_zscore", ascending=False)
        st.dataframe(anomalous, use_container_width=True)

    except FileNotFoundError:
        st.error("Graph metrics not found. Run the pipeline first: `python run.py`")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: PATIENT CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️ 6. Check My Doctor (For Patients)":
    st.title("🛡️ Check My Doctor (For Patients)")
    st.info("💡 **What is this?** Are you a patient who just received a suspicious medical bill? Enter your doctor's NPI (National Provider Identifier) and the amount they charged you to see if they are engaging in suspicious billing practices or are banned by the US Government!")
    
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Claim Details")
        # Pre-fill with a known NPI for demo purposes (preferably one that's a fraud case if available)
        fraud_npis_list = providers[providers["is_fraud"] == 1]["provider_id"].tolist()
        demo_npi = fraud_npis_list[0] if len(fraud_npis_list) > 0 else providers["provider_id"].iloc[0]
        
        provider_id = st.text_input("Doctor's NPI (10-digit number)", value=str(demo_npi))
        claim_amount = st.number_input("Amount Billed to you ($)", min_value=0.0, max_value=1000000.0, value=1500.0)
        
        check_btn = st.button("🔍 Check for Fraud", type="primary", use_container_width=True)
        
    with col2:
        if check_btn:
            st.subheader("Analysis Results")
            if provider_id not in providers["provider_id"].values:
                st.warning("Doctor NPI not found in our 2022 Medicare dataset.")
            else:
                prov_data = providers[providers["provider_id"] == provider_id].iloc[0]
                
                # Check LEIE (Is Fraud)
                if prov_data["is_fraud"] == 1:
                    st.error("🚨 **CRITICAL ALERT:** This doctor is on the US Government's Official **Banned List (LEIE)** for committing Medicare fraud, kickbacks, or patient abuse!")
                else:
                    st.success("✅ This doctor is NOT on the official government banned list.")
                
                # Check historical averages
                prov_claims = claims[claims["provider_id"] == provider_id]
                if len(prov_claims) > 0:
                    avg_cost = prov_claims["claim_amount"].mean()
                    st.markdown(f"**Doctor's Specialty:** {prov_data['specialty']}")
                    st.markdown(f"**Average amount this doctor usually charges:** ${avg_cost:,.2f}")
                    
                    if claim_amount > avg_cost * 3:
                        st.warning(f"⚠️ **UPCODING ALERT:** Your bill of **${claim_amount:,.2f}** is more than 3X higher than what this doctor normally charges Medicare. You should consider contacting your insurance provider to dispute this charge!")
                    elif claim_amount > avg_cost * 1.5:
                        st.info("ℹ️ **Notice:** Your bill is higher than this doctor's average, but within normal variance.")
                    else:
                        st.success("✅ Your bill looks completely normal compared to this doctor's historical averages.")
                    
                    # Network analysis check
                    try:
                        with open(DATA_DIR / "graph_metrics.pkl", "rb") as f:
                            graph_metrics = pickle.load(f)
                        community_map = graph_metrics["community"]
                        comm_id = community_map.get(provider_id, -1)
                        if comm_id != -1:
                            st.markdown(f"**Network Analysis:** This doctor belongs to patient-sharing **Network #{comm_id}**.")
                    except Exception:
                        pass
