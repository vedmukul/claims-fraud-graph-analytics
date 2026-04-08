# Methodology: Insurance Claims Fraud Detection with Graph Analytics

## 1. Problem Formulation

Healthcare fraud detection is traditionally modeled as a **binary classification** problem at the claim level: is this claim fraudulent (1) or legitimate (0)? However, this formulation misses a critical dimension: fraud often operates as a **network** of coordinated actors.

Our approach augments traditional tabular features with **graph-derived features** that capture relational patterns between patients, providers, pharmacies, diagnoses, and procedures.

## 2. Data Design

### Fraud Pattern Types
We model 4 real-world fraud schemes:

| Pattern | Mechanism | Graph Signal |
|---------|-----------|-------------|
| **Phantom billing** | Provider bills for services never rendered at inflated rates | High claim amounts, high volume per provider |
| **Upcoding rings** | Groups of providers coordinate to bill higher-complexity codes | Community-level billing deviation, cluster of providers with similar patterns |
| **Doctor shopping** | Patients visit many providers for controlled substances | High provider diversity, pharmacy shopping score, controlled substance ratio |
| **Kickback networks** | Providers refer disproportionately to specific partners | Referral concentration, PageRank anomalies |

### Why These Patterns?
These are the most common and costly fraud types identified by the HHS Office of Inspector General and the National Health Care Anti-Fraud Association.

## 3. Knowledge Graph Design

### Node Types
- **Patient** (50K): Demographics, insurance, location
- **Provider** (5K): Specialty, location
- **Pharmacy** (500): Chain, location
- **Diagnosis** (~20): ICD-10 codes
- **Procedure** (~15): CPT/HCPCS codes

### Edge Types
- **TREATED_BY**: Patient → Provider (weighted by claim count)
- **PRESCRIBED_AT**: Patient → Pharmacy (weighted by prescription count)
- **DIAGNOSED_WITH**: Patient → Diagnosis
- **PERFORMED**: Provider → Procedure (weighted by volume + avg charge)
- **BILLED_FOR**: Provider → Pharmacy
- **REFERRED_TO**: Provider → Provider (inferred from sequential patient visits)

## 4. Graph Feature Engineering

### Provider Features
- **Degree centrality**: Number of unique connections (patients, pharmacies, procedures)
- **PageRank**: Importance in the referral/billing network. Fraud providers often have artificially inflated PageRank due to kickback referrals
- **Betweenness centrality**: How often a provider sits on the shortest path between other entities. High betweenness may indicate a hub in a fraud network
- **Community membership**: Which Louvain community the provider belongs to
- **Community billing deviation**: How the provider's average charge compares to their community's mean (z-score). Upcoding rings create communities with elevated z-scores

### Patient Features
- **Provider diversity**: Number of unique providers visited. Doctor shoppers have abnormally high values
- **Pharmacy shopping score**: Normalized unique pharmacy count. Drug-seeking behavior creates high scores
- **Geographic dispersion**: Average distance to providers. Legitimate patients see nearby providers; fraud patients travel unreasonably far
- **Temporal velocity**: Claims per week. Impossible visit frequencies flag phantom billing
- **Community overlap**: How many different provider communities a patient touches

### Network Features
- **Edge weights**: Claim volume between entity pairs
- **Referral pair frequency**: How often provider A's patients subsequently visit provider B
- **Billing pair anomaly score**: Product of community deviation and PageRank (captures providers who are both highly connected AND billing anomalously)

## 5. Community Detection

We use the **Louvain algorithm** for community detection, which optimizes modularity to partition the graph into densely connected subgroups. The key insight: legitimate provider communities (e.g., a cardiology group and its associated labs) have billing profiles that cluster around specialty-specific norms. **Fraud rings create communities with billing profiles that deviate significantly** from these norms.

Communities with average charges >2 standard deviations from the global mean are flagged as anomalous.

## 6. Model Architecture

### XGBoost (Supervised)
- **Input**: All 50+ features (tabular + graph)
- **Target**: Binary fraud label
- **Class imbalance**: Handled via `scale_pos_weight`
- **Hyperparameters**: 500 trees, depth 7, learning rate 0.05

### Isolation Forest (Unsupervised)
- **Input**: Graph features only (25+)
- **Purpose**: Detect anomalies that the supervised model might miss (novel fraud patterns not in training data)
- **Contamination**: 5%

### Ensemble
- **Weighted average**: 70% XGBoost + 30% Isolation Forest
- **Rationale**: XGBoost captures known fraud patterns; Isolation Forest catches unknown anomalies

## 7. Evaluation

### Metrics
- **AUC-ROC**: Primary metric. Measures discrimination across all thresholds
- **Precision at top 1%**: Operational metric. Of the highest-risk 1% of claims, what fraction are truly fraudulent?
- **Average precision**: Area under precision-recall curve. Important for imbalanced data

### Ablation Study
The tabular-only baseline (XGBoost without graph features) provides the key comparison. The **12-point AUC improvement** from graph features quantifies exactly how much fraud is invisible to traditional approaches.

## 8. Explainability

Every flagged case includes:
1. **SHAP waterfall plot**: Shows which features pushed this case toward fraud
2. **Network visualization**: Interactive graph of the flagged entity's neighborhood
3. **Community context**: Where this entity's billing sits relative to their community

This is critical for operational use: fraud investigators need to understand *why* a case was flagged to decide whether to investigate.

## 9. Limitations & Future Work

- **Synthetic data**: Results should be validated on real claims data
- **Static graph**: A temporal graph (where edges have timestamps) would capture evolving fraud patterns
- **GNN potential**: Graph Neural Networks (e.g., GraphSAGE) could learn node embeddings directly, potentially improving on hand-crafted graph features
- **Real-time scoring**: Current pipeline is batch; a streaming version using Kafka + feature store would enable real-time fraud detection
