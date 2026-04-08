<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NetworkX-Graph%20Analytics-FF6F00?style=for-the-badge&logo=graphql&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Ensemble%20ML-006400?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8B0000?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🔍 Insurance Claims Fraud Detection
### Graph Analytics · Ensemble Machine Learning · Explainable AI

> **Detecting healthcare fraud networks using knowledge graphs, community detection, and ensemble ML — built on real CMS Medicare data with OIG LEIE fraud labels.**

---

## 🎯 The Problem

Healthcare fraud costs the U.S. **$100 billion+ annually**. Traditional fraud detection relies on rule-based systems or single-record anomaly detection. But sophisticated fraud operates as **networks**:

| Fraud Pattern | Description | Why Traditional Methods Fail |
|---|---|---|
| 🏥 **Phantom Billing** | Providers billing for services never rendered | Looks like normal claims individually |
| 💊 **Doctor Shopping** | Patients visiting 10+ providers for controlled substances | Requires cross-provider analysis |
| 📈 **Upcoding Rings** | Groups of providers systematically charging higher-complexity codes | Only visible when comparing peer groups |
| 🤝 **Kickback Networks** | Providers with abnormal referral concentration to specific partners | Invisible without referral network analysis |
| ⏰ **Burst Billing** | Pop-up clinic billing hundreds of claims in a 2-week window | Requires temporal pattern analysis |

These patterns are **invisible to tabular analysis** but become obvious when you analyze the **relationships between entities**.

---

## 💡 The Solution

This project constructs a **knowledge graph** from real CMS Medicare claims data, applies **graph-based feature engineering** and **community detection** to identify suspicious networks, and combines graph signals with traditional features in an **ensemble model** with full **SHAP explainability**.

```
 ┌──────────────────────┐     ┌───────────────────────┐     ┌──────────────────────────┐
 │   CMS Medicare API   │────▶│   Knowledge Graph     │────▶│   Feature Engineering    │
 │   + OIG LEIE Labels  │     │   (NetworkX)          │     │   (Tabular + Graph)      │
 │   250K+ providers    │     │   200K+ nodes         │     │   55 features            │
 │   2.3M claims        │     │   3M+ edges           │     │   23 graph-derived       │
 └──────────────────────┘     └───────────────────────┘     └────────────┬─────────────┘
                                                                         │
                          ┌────────────────────────┐     ┌───────────────▼───────────────┐
                          │   Streamlit Dashboard   │◀───│   Ensemble Model              │
                          │   6 Interactive Pages   │    │   XGBoost + Isolation Forest  │
                          │   SHAP + Graph Viz      │    │   + SHAP Explainability       │
                          │   Patient Checker       │    │   1.8M train / 462K test      │
                          └────────────────────────┘    └───────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker-compose up --build
# Dashboard → http://localhost:8501
```

### Option 2: Local
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (fetches data, builds graph, trains model)
python run.py

# Launch the interactive dashboard
streamlit run dashboards/app.py
```

> **Note:** The first run fetches ~250K records from the CMS API and builds a 695MB knowledge graph. Subsequent runs load cached data from the `data/` directory automatically.

---

## 🏗️ Project Structure

```
claims-fraud-graph-analytics/
│
├── run.py                              # 🎯 Pipeline orchestrator — runs all 5 stages
│
├── src/
│   ├── config.py                       # ⚙️  Central configuration & hyperparameters
│   ├── fetch_real_cms_data.py          # 📡 Fetches real CMS Medicare + OIG LEIE data
│   ├── generate_synthetic_data.py      # 🧪 Alternative synthetic data generator
│   ├── build_graph.py                  # 🕸️  Knowledge graph construction + metrics
│   ├── feature_engineering.py          # 🔧 55 features (tabular + graph extraction)
│   └── train_model.py                  # 🤖 XGBoost + Isolation Forest + SHAP
│
├── dashboards/
│   └── app.py                          # 📊 6-page Streamlit investigation dashboard
│
├── data/                               # 📁 Generated data artifacts (gitignored)
├── models/                             # 📁 Trained model artifacts (gitignored)
├── tests/
│   └── test_pipeline.py                # ✅ Unit tests
│
├── docs/
│   └── methodology.md                  # 📖 Detailed methodology documentation
│
├── Dockerfile                          # 🐳 Container configuration
├── docker-compose.yml                  # 🐳 One-command deployment
├── requirements.txt                    # 📦 Python dependencies
└── .gitignore
```

---

## 📐 Pipeline Deep Dive

### Stage 1 — Data Acquisition

Fetches **real Medicare Part D Prescriber data** from the [CMS Open Data API](https://data.cms.gov/) in 50K-record chunks (250K total). Fraud labels are sourced from the **OIG LEIE** (Office of Inspector General — List of Excluded Individuals/Entities), the U.S. government's official database of healthcare providers convicted of fraud, patient abuse, or felonies.

Patient-provider relationships are generated with **geospatial anomaly injection**:
- **Normal patients**: placed within ~15 miles of their provider
- **Fraud patients** (30% chance): placed **1,000+ miles away** — simulating impossible travel patterns
- **Burst billing** (40% of fraud providers): 80% of claims concentrated in a 14-day window

### Stage 2 — Knowledge Graph Construction

A **heterogeneous graph** with 5 node types and 6 relationship types:

```
   Patient ──TREATED_BY──▶ Provider
   Patient ──PRESCRIBED_AT──▶ Pharmacy
   Patient ──DIAGNOSED_WITH──▶ Diagnosis
   Provider ──PERFORMED──▶ Procedure
   Provider ──BILLED_FOR──▶ Pharmacy
   Provider ──REFERRED_TO──▶ Provider  (inferred from sequential visits)
```

Edges are weighted by claim frequency, with referral edges inferred from temporal sequences of patient visits.

### Stage 3 — Graph Metrics

| Metric | Algorithm | What It Detects |
|--------|-----------|-----------------|
| **PageRank** | Weighted PageRank (α=0.85) | Providers with outsized influence in referral networks |
| **Degree Centrality** | NetworkX | Providers connected to unusually many entities |
| **Betweenness Centrality** | Sampled betweenness | Providers bridging otherwise disconnected communities |
| **Community Detection** | Louvain algorithm | Provider clusters that may represent coordinated fraud rings |

### Stage 4 — Feature Engineering (55 Features)

<table>
<tr><th>Category</th><th>Features</th><th>Fraud Signal</th></tr>
<tr>
  <td><strong>Provider Graph</strong> (12)</td>
  <td>PageRank, degree/betweenness centrality, community ID/size/avg charge/deviation, referral concentration, geographic dispersion, unique patients/pharmacies</td>
  <td>Inflated referral schemes, coordinated upcoding, impossible geography</td>
</tr>
<tr>
  <td><strong>Patient Graph</strong> (7)</td>
  <td>Pharmacy shopping score, geographic dispersion, temporal velocity, provider diversity, community overlap</td>
  <td>Doctor shopping, controlled substance abuse, unreasonable travel</td>
</tr>
<tr>
  <td><strong>Pharmacy Graph</strong> (4)</td>
  <td>Unique providers, controlled substance ratio, patient volume, geographic spread</td>
  <td>Pill mill detection, abnormal prescription volume</td>
</tr>
<tr>
  <td><strong>Provider Tabular</strong> (12)</td>
  <td>Avg/std charge, claim volume, burst billing ratio, weekend/night ratios, complexity</td>
  <td>Billing anomalies, after-hours fraud, upcoding</td>
</tr>
<tr>
  <td><strong>Patient Tabular</strong> (12)</td>
  <td>Claim count, avg/std amount, claims/week, controlled substance ratio, age</td>
  <td>Excessive utilization, substance abuse patterns</td>
</tr>
<tr>
  <td><strong>Edge Features</strong> (4)</td>
  <td>Patient-provider weight, provider-pharmacy weight, referral pair frequency, billing pair anomaly score</td>
  <td>Unusual relationship strength, kickback patterns</td>
</tr>
</table>

### Stage 5 — Ensemble Model + Explainability

| Component | Role | Configuration |
|-----------|------|---------------|
| **XGBoost** | Supervised classifier | 500 trees, depth 7, LR 0.05, `scale_pos_weight` for imbalance |
| **Isolation Forest** | Unsupervised anomaly detector (graph features only) | 200 estimators, 5% contamination |
| **Ensemble** | Weighted combination | 70% XGBoost + 30% Isolation Forest |
| **SHAP** | Per-case explainability | TreeExplainer on 5,000 test samples |
| **Baseline** | Tabular-only XGBoost for comparison | Same hyperparameters, no graph features |

---

## 📊 Dashboard Pages

| # | Page | What It Shows |
|---|------|---------------|
| 1 | **📊 Overall AI Performance** | KPI cards, AUC comparison bar chart, top 15 feature importance, fraud score distributions |
| 2 | **🚨 Suspicious Doctors Queue** | Ranked table of highest-risk claims with sortable columns and precision metrics |
| 3 | **🧠 Why was it flagged?** | SHAP beeswarm (global) + waterfall (per-case) plots explaining AI decisions |
| 4 | **🕸️ Doctor & Patient Network** | Interactive Pyvis graph visualization of fraud provider neighborhoods |
| 5 | **🏘️ Fraud Ring Detection** | Louvain community analysis, anomalous community flagging (>2σ billing deviation) |
| 6 | **🛡️ Check My Doctor** | Patient-facing NPI lookup with upcoding alerts and LEIE ban check |

---

## 🔬 Methodology Highlights

1. **Real data foundation**: Built on actual CMS Medicare Part D prescriber records, not purely synthetic data
2. **Graph-derived features**: 23 features extracted from a 200K+ node knowledge graph that capture relational patterns invisible to flat tables
3. **Louvain community detection**: Groups providers into communities based on shared patient networks — anomalous communities flagged as potential fraud rings
4. **Geospatial anomaly injection**: Simulates "impossible travel" fraud patterns where patients are thousands of miles from their treating provider
5. **Burst billing detection**: Identifies providers with temporal claim concentration anomalies (pop-up clinic pattern)
6. **Full SHAP explainability**: Every flagged case includes feature-level attribution — critical for investigator trust and regulatory compliance

---

## 🛠️ Technologies

| Category | Stack |
|----------|-------|
| **Graph Analytics** | NetworkX, python-louvain, Pyvis |
| **Machine Learning** | XGBoost, Scikit-learn, Isolation Forest, imbalanced-learn |
| **Explainability** | SHAP |
| **Data** | Pandas, NumPy, PyArrow, SciPy |
| **Visualization** | Plotly, Matplotlib, Seaborn, Streamlit |
| **Infrastructure** | Docker, Docker Compose |
| **Data Sources** | CMS Open Data API, OIG LEIE Exclusion Database |

---

## 📋 Requirements

- Python 3.10+
- ~4GB RAM (graph construction is memory-intensive)
- Internet connection for first run (CMS API data fetch)
- See [requirements.txt](requirements.txt) for full dependency list

---

## 📄 License

MIT

---

<p align="center">
  <em>Built to demonstrate how graph analytics can uncover fraud networks invisible to traditional methods.</em>
</p>
