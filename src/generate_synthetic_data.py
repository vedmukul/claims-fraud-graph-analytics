"""
Synthetic Healthcare Claims Data Generator
Generates realistic claims data with embedded fraud patterns:
  - Phantom billing (providers billing for services never rendered)
  - Upcoding rings (coordinated higher-complexity billing)
  - Doctor shopping (patients visiting many providers for controlled substances)
  - Kickback networks (abnormal referral concentration)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


# ─── Reference Data ────────────────────────────────────────────────────────────

SPECIALTIES = [
    "Family Medicine", "Internal Medicine", "Cardiology", "Orthopedics",
    "Neurology", "Psychiatry", "Dermatology", "Oncology", "Radiology",
    "Emergency Medicine", "Pain Management", "General Surgery",
    "Pulmonology", "Gastroenterology", "Endocrinology",
]

DIAGNOSIS_CODES = {
    "E11.9": "Type 2 diabetes", "I10": "Hypertension", "J06.9": "Upper respiratory infection",
    "M54.5": "Low back pain", "F32.9": "Major depressive disorder", "J44.1": "COPD exacerbation",
    "I25.10": "Coronary artery disease", "E78.5": "Hyperlipidemia", "N39.0": "Urinary tract infection",
    "K21.0": "GERD", "G43.909": "Migraine", "M79.3": "Panniculitis", "R10.9": "Abdominal pain",
    "F41.1": "Generalized anxiety", "T40.2X1A": "Opioid poisoning", "F11.20": "Opioid dependence",
    "F10.20": "Alcohol dependence", "R51": "Headache", "M25.50": "Joint pain",
    "Z79.891": "Long-term opioid use",
}

CONTROLLED_DIAGNOSIS = ["T40.2X1A", "F11.20", "F10.20", "Z79.891"]

PROCEDURES = {
    "99213": ("Office visit level 3", 85, 150), "99214": ("Office visit level 4", 120, 250),
    "99215": ("Office visit level 5", 180, 350), "99281": ("ED visit level 1", 100, 200),
    "99285": ("ED visit level 5", 500, 1200), "73721": ("MRI knee", 400, 2500),
    "71046": ("Chest X-ray", 50, 200), "93000": ("ECG", 30, 100),
    "80053": ("Comprehensive metabolic panel", 20, 80), "85025": ("CBC", 15, 50),
    "90834": ("Psychotherapy 45min", 100, 200), "90837": ("Psychotherapy 60min", 130, 280),
    "20610": ("Joint injection", 100, 400), "J2310": ("Naloxone injection", 40, 150),
    "96372": ("Therapeutic injection", 25, 80),
}

STATES = ["IL", "IN", "WI", "MI", "OH", "MN", "IA", "MO", "KY", "PA", "NY", "CA", "TX", "FL", "GA"]


# ─── Entity Generators ─────────────────────────────────────────────────────────

def generate_patients(n):
    logger.info(f"Generating {n:,} patients...")
    ages = np.clip(np.random.normal(55, 18, n), 18, 95).astype(int)
    genders = np.random.choice(["M", "F"], n, p=[0.48, 0.52])
    states = np.random.choice(STATES, n, p=[0.15, 0.08, 0.06, 0.07, 0.07, 0.05, 0.04, 0.05, 0.04, 0.06, 0.08, 0.1, 0.08, 0.04, 0.03])
    lats = np.random.normal(41.8, 2.5, n)
    lons = np.random.normal(-87.6, 3.0, n)
    insurance = np.random.choice(["Medicare", "Medicaid", "Commercial", "Self-Pay"], n, p=[0.35, 0.20, 0.40, 0.05])
    return pd.DataFrame({
        "patient_id": [f"PAT_{i:06d}" for i in range(n)],
        "age": ages, "gender": genders, "state": states,
        "lat": lats, "lon": lons, "insurance_type": insurance,
        "is_fraud": False,
    })


def generate_providers(n):
    logger.info(f"Generating {n:,} providers...")
    specialties = np.random.choice(SPECIALTIES, n)
    states = np.random.choice(STATES, n)
    lats = np.random.normal(41.8, 2.5, n)
    lons = np.random.normal(-87.6, 3.0, n)
    return pd.DataFrame({
        "provider_id": [f"PRV_{i:05d}" for i in range(n)],
        "specialty": specialties, "state": states,
        "lat": lats, "lon": lons,
        "is_fraud": False,
    })


def generate_pharmacies(n):
    logger.info(f"Generating {n:,} pharmacies...")
    chain = np.random.choice(["CVS", "Walgreens", "Rite Aid", "Walmart", "Independent"], n, p=[0.25, 0.25, 0.1, 0.15, 0.25])
    states = np.random.choice(STATES, n)
    lats = np.random.normal(41.8, 2.5, n)
    lons = np.random.normal(-87.6, 3.0, n)
    return pd.DataFrame({
        "pharmacy_id": [f"PHR_{i:04d}" for i in range(n)],
        "chain": chain, "state": states, "lat": lats, "lon": lons,
    })


# ─── Claims Generator ──────────────────────────────────────────────────────────

def generate_base_claims(n, patients, providers, pharmacies):
    logger.info(f"Generating {n:,} base claims...")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    proc_codes = list(PROCEDURES.keys())
    diag_codes = list(DIAGNOSIS_CODES.keys())

    claims = pd.DataFrame({
        "claim_id": [f"CLM_{i:08d}" for i in range(n)],
        "patient_id": np.random.choice(patients["patient_id"].values, n),
        "provider_id": np.random.choice(providers["provider_id"].values, n),
        "pharmacy_id": np.random.choice(pharmacies["pharmacy_id"].values, n),
        "procedure_code": np.random.choice(proc_codes, n),
        "diagnosis_code": np.random.choice(diag_codes, n),
        "claim_date": [start_date + timedelta(days=int(np.random.randint(0, date_range))) for _ in range(n)],
        "is_fraud": False,
    })

    # Calculate amounts based on procedure
    amounts = []
    complexities = []
    for proc in claims["procedure_code"]:
        low, high = PROCEDURES[proc][1], PROCEDURES[proc][2]
        amounts.append(round(np.random.uniform(low, high), 2))
        complexities.append(round((high - low) / low, 2))
    claims["claim_amount"] = amounts
    claims["procedure_complexity"] = complexities

    # Add time features
    claims["day_of_week"] = claims["claim_date"].apply(lambda x: x.weekday())
    claims["hour"] = np.random.choice(range(6, 22), n)
    claims["is_weekend"] = claims["day_of_week"].isin([5, 6]).astype(int)
    claims["is_night"] = ((claims["hour"] < 8) | (claims["hour"] > 18)).astype(int)

    return claims


# ─── Fraud Pattern Injection ───────────────────────────────────────────────────

def inject_phantom_billing(claims, providers, n_fraud_providers=50):
    """Providers billing for services with inflated amounts and impossible volumes."""
    logger.info(f"Injecting phantom billing fraud ({n_fraud_providers} providers)...")
    fraud_providers = np.random.choice(providers["provider_id"].values, n_fraud_providers, replace=False)
    providers.loc[providers["provider_id"].isin(fraud_providers), "is_fraud"] = True

    mask = claims["provider_id"].isin(fraud_providers)
    # Inflate amounts by 2-5x
    claims.loc[mask, "claim_amount"] *= np.random.uniform(2.0, 5.0, mask.sum())
    claims.loc[mask, "claim_amount"] = claims.loc[mask, "claim_amount"].round(2)
    # Mark as fraud
    claims.loc[mask, "is_fraud"] = True
    return claims, providers


def inject_upcoding_ring(claims, providers, n_rings=3, providers_per_ring=15):
    """Groups of providers systematically using higher-complexity procedure codes."""
    logger.info(f"Injecting {n_rings} upcoding rings...")
    upcode_map = {"99213": "99215", "99214": "99215", "99281": "99285"}

    for ring in range(n_rings):
        ring_providers = np.random.choice(
            providers[~providers["is_fraud"]]["provider_id"].values,
            providers_per_ring, replace=False
        )
        providers.loc[providers["provider_id"].isin(ring_providers), "is_fraud"] = True

        for orig, upcoded in upcode_map.items():
            mask = (claims["provider_id"].isin(ring_providers)) & (claims["procedure_code"] == orig)
            # 70% of eligible claims get upcoded
            upcode_mask = mask & (np.random.random(len(claims)) < 0.7)
            claims.loc[upcode_mask, "procedure_code"] = upcoded
            new_low, new_high = PROCEDURES[upcoded][1], PROCEDURES[upcoded][2]
            claims.loc[upcode_mask, "claim_amount"] = np.random.uniform(new_low, new_high, upcode_mask.sum()).round(2)
            claims.loc[upcode_mask, "is_fraud"] = True

    return claims, providers


def inject_doctor_shopping(claims, patients, n_shoppers=500):
    """Patients visiting many different providers for controlled substances."""
    logger.info(f"Injecting doctor shopping fraud ({n_shoppers} patients)...")
    shoppers = np.random.choice(patients["patient_id"].values, n_shoppers, replace=False)
    patients.loc[patients["patient_id"].isin(shoppers), "is_fraud"] = True

    mask = claims["patient_id"].isin(shoppers)
    # Force controlled substance diagnoses
    claims.loc[mask, "diagnosis_code"] = np.random.choice(CONTROLLED_DIAGNOSIS, mask.sum())
    claims.loc[mask, "is_fraud"] = True

    # Add extra claims for shoppers (they visit more often)
    extra_claims = claims[mask].sample(n=min(50000, mask.sum()), replace=True).copy()
    extra_claims["claim_id"] = [f"CLM_EXTRA_{i:06d}" for i in range(len(extra_claims))]
    extra_claims["provider_id"] = np.random.choice(
        claims["provider_id"].unique(), len(extra_claims)
    )
    extra_claims["pharmacy_id"] = np.random.choice(
        claims["pharmacy_id"].unique(), len(extra_claims)
    )
    extra_claims["is_fraud"] = True

    claims = pd.concat([claims, extra_claims], ignore_index=True)
    return claims, patients


def inject_kickback_network(claims, providers, n_hubs=10, n_referral_targets=3):
    """Providers referring disproportionately to a small set of other providers."""
    logger.info(f"Injecting kickback networks ({n_hubs} hubs)...")
    available = providers[~providers["is_fraud"]]["provider_id"].values

    for hub_idx in range(n_hubs):
        hub = np.random.choice(available, 1)[0]
        targets = np.random.choice(available[available != hub], n_referral_targets, replace=False)
        providers.loc[providers["provider_id"] == hub, "is_fraud"] = True
        providers.loc[providers["provider_id"].isin(targets), "is_fraud"] = True

        # Create referral claims: patients seen by hub then seen by target
        hub_claims = claims[claims["provider_id"] == hub].sample(n=min(200, len(claims[claims["provider_id"] == hub])), replace=True)
        for target in targets:
            referral_claims = hub_claims.copy()
            referral_claims["claim_id"] = [f"CLM_REF_{hub_idx}_{i:05d}" for i in range(len(referral_claims))]
            referral_claims["provider_id"] = target
            referral_claims["claim_date"] = referral_claims["claim_date"] + timedelta(days=np.random.randint(1, 14))
            referral_claims["claim_amount"] *= np.random.uniform(1.5, 3.0, len(referral_claims))
            referral_claims["claim_amount"] = referral_claims["claim_amount"].round(2)
            referral_claims["is_fraud"] = True
            claims = pd.concat([claims, referral_claims], ignore_index=True)

    return claims, providers


# ─── Main Generation Pipeline ──────────────────────────────────────────────────

def generate_all():
    """Generate all synthetic data with fraud patterns."""
    logger.info("=" * 60)
    logger.info("STARTING SYNTHETIC DATA GENERATION")
    logger.info("=" * 60)

    # Generate base entities
    patients = generate_patients(N_PATIENTS)
    providers = generate_providers(N_PROVIDERS)
    pharmacies = generate_pharmacies(N_PHARMACIES)

    # Generate base claims
    claims = generate_base_claims(N_CLAIMS, patients, providers, pharmacies)

    # Inject fraud patterns
    claims, providers = inject_phantom_billing(claims, providers)
    claims, providers = inject_upcoding_ring(claims, providers)
    claims, patients = inject_doctor_shopping(claims, patients)
    claims, providers = inject_kickback_network(claims, providers)

    # Summary stats
    logger.info("=" * 60)
    logger.info(f"Total claims: {len(claims):,}")
    logger.info(f"Fraud claims: {claims['is_fraud'].sum():,} ({claims['is_fraud'].mean():.1%})")
    logger.info(f"Fraud providers: {providers['is_fraud'].sum():,} ({providers['is_fraud'].mean():.1%})")
    logger.info(f"Fraud patients: {patients['is_fraud'].sum():,} ({patients['is_fraud'].mean():.1%})")
    logger.info("=" * 60)

    # Save
    claims.to_parquet(DATA_DIR / "claims.parquet", index=False)
    patients.to_parquet(DATA_DIR / "patients.parquet", index=False)
    providers.to_parquet(DATA_DIR / "providers.parquet", index=False)
    pharmacies.to_parquet(DATA_DIR / "pharmacies.parquet", index=False)

    logger.info(f"Data saved to {DATA_DIR}")
    return claims, patients, providers, pharmacies


if __name__ == "__main__":
    generate_all()
