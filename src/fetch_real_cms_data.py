import pandas as pd
import requests
import logging
import datetime
from pathlib import Path
import numpy as np
import uuid

logger = logging.getLogger(__name__)

def generate_all():
    logger.info("Fetching real Medicare Part D Prescriber data from CMS API...")
    logger.info("Fetching real Medicare Part D Prescriber data from CMS API in chunks...")
    dfs = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for offset in range(0, 250000, 50000):
        url = f"https://data.cms.gov/data-api/v1/dataset/14d8e8a9-7e9b-4370-a044-bf97c46b4b44/data?size=50000&offset={offset}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        dfs.append(pd.DataFrame(response.json()))
        logger.info(f"  Fetched chunk offset={offset}...")
    cms_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total providers fetched: {len(cms_df)}")
    
    logger.info("Fetching OIG LEIE Fraud database...")
    leie_df = pd.DataFrame()
    try:
        leie_url = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
        leie_response = requests.get(leie_url, headers=headers)
        if leie_response.status_code == 200:
            from io import StringIO
            leie_df = pd.read_csv(StringIO(leie_response.text), low_memory=False)
            logger.info("Successfully fetched LEIE database.")
        else:
            logger.warning(f"Failed to fetch LEIE, HTTP {leie_response.status_code}")
    except Exception as e:
        logger.warning(f"Failed to fetch LEIE: {e}")
        
    # 1. Providers
    cms_df["PRSCRBR_NPI"] = pd.to_numeric(cms_df["PRSCRBR_NPI"], errors="coerce")
    cms_df = cms_df.dropna(subset=["PRSCRBR_NPI"])
    
    fraud_npis = set()
    if not leie_df.empty and "NPI" in leie_df.columns:
        fraud_npis = set(leie_df[leie_df["NPI"] != 0]["NPI"])
        
    if len(fraud_npis) == 0:
        logger.warning("Mocking fraud labels because LEIE NPIs could not be loaded")
        fraud_npis = set(cms_df["PRSCRBR_NPI"].sample(frac=0.03, random_state=42))
        
    cms_df["is_fraud"] = cms_df["PRSCRBR_NPI"].isin(fraud_npis).astype(int)
    
    providers = cms_df[["PRSCRBR_NPI", "Prscrbr_Type", "is_fraud", "Prscrbr_State_Abrvtn"]].copy()
    providers.columns = ["provider_id", "specialty", "is_fraud", "state"]
    
    np.random.seed(42)
    providers["lat"] = np.random.uniform(25, 49, len(providers))
    providers["lon"] = np.random.uniform(-125, -66, len(providers))
    providers = providers.drop_duplicates(subset=["provider_id"])
    
    # 2. Patients & 4. Claims (Geospatial Simulation)
    logger.info("Building Patient-Provider Geospatial Graph...")
    claims_list = []
    patients_dict = {}
    
    # Pre-generate provider lat/lon map for fast lookup
    prov_latlon = providers.set_index("provider_id")[["lat", "lon"]].to_dict("index")
    
    patient_counter = 0
    for _, row in cms_df.iterrows():
        pid = row["PRSCRBR_NPI"]
        is_fraud = row["is_fraud"]
        plat, plon = prov_latlon[pid]["lat"], prov_latlon[pid]["lon"]
        
        # Burst Billing Anomaly Injection
        is_burst_biller = False
        anchor_month = None
        if is_fraud == 1 and np.random.rand() < 0.40:
            is_burst_biller = True
            anchor_month = np.random.randint(1, 12)
            
        def add_claims(count_col, prefix):
            nonlocal patient_counter
            count = pd.to_numeric(row.get(count_col, 0), errors="coerce")
            if pd.isna(count) or count <= 0: return
            for _ in range(int(min(count, 50))):
                patient_id = f"PAT_{patient_counter}"
                patient_counter += 1
                
                # Geospatial Anomaly Injection (Impossible Travel)
                is_impossible_travel = False
                if is_fraud == 1 and np.random.rand() < 0.30:
                    # Fraudster patients live > 1000 miles away
                    pat_lat = plat + np.random.choice([15.0, -15.0]) + np.random.uniform(-2, 2)
                    pat_lon = plon + np.random.choice([20.0, -20.0]) + np.random.uniform(-5, 5)
                    is_impossible_travel = True
                else:
                    # Normal patients live very close (within ~15 miles)
                    pat_lat = plat + np.random.uniform(-0.2, 0.2)
                    pat_lon = plon + np.random.uniform(-0.2, 0.2)
                
                patients_dict[patient_id] = {
                    "patient_id": patient_id,
                    "age": np.random.randint(20, 90),
                    "gender": np.random.choice(["M", "F"]),
                    "lat": pat_lat,
                    "lon": pat_lon,
                    "state": row["Prscrbr_State_Abrvtn"],
                    "is_fraud": 0,
                    "insurance_type": "Medicare"
                }

                if is_burst_biller and np.random.rand() < 0.8:
                    # 80% of claims forced into a tiny 14-day pop-up clinic window
                    day = np.random.randint(1, 15)
                    date = datetime.date(2022, anchor_month, day)
                else:
                    date = datetime.date(2022, np.random.randint(1, 13), np.random.randint(1, 28))
                claims_list.append({
                    "claim_id": str(uuid.uuid4()),
                    "patient_id": patient_id,
                    "provider_id": pid,
                    "pharmacy_id": "PHARM_1",
                    "claim_date": date,
                    "claim_amount": np.random.uniform(10, 500),
                    "procedure_complexity": np.random.uniform(1, 4),
                    "is_fraud": is_fraud,
                    "diagnosis_code": "D" + str(np.random.randint(100, 200)),
                    "procedure_code": "P" + str(np.random.randint(1000, 2000)),
                    # The Drug class can be stored in a separate column or implied
                    "drug_class": prefix
                })
                
        add_claims("Opioid_Tot_Clms", "Opioid")
        add_claims("Antbtc_Tot_Clms", "Antibiotic")
        add_claims("Brnd_Tot_Clms", "Brand")
        add_claims("Gnrc_Tot_Clms", "Generic")
        
    patients = pd.DataFrame.from_dict(patients_dict, orient='index')
    
    # 3. Pharmacies
    pharmacies = pd.DataFrame({
        "pharmacy_id": ["PHARM_1", "PHARM_2"],
        "lat": [40.0, 41.0],
        "lon": [-80.0, -81.0],
        "state": ["NY", "PA"],
        "is_fraud": [0, 0],
        "chain": [1, 0]
    })
    
    claims_df = pd.DataFrame(claims_list)
    
    if claims_df.empty:
        raise ValueError("Failed to generate claims")

    claims_df["claim_date"] = pd.to_datetime(claims_df["claim_date"])
    claims_df["day_of_week"] = claims_df["claim_date"].dt.dayofweek
    claims_df["is_weekend"] = claims_df["day_of_week"].isin([5, 6]).astype(int)
    claims_df["is_night"] = 0
    claims_df["claim_amount_zscore"] = (claims_df["claim_amount"] - claims_df["claim_amount"].mean()) / claims_df["claim_amount"].std()
    claims_df["claim_amount_percentile"] = claims_df["claim_amount"].rank(pct=True)
    
    logger.info(f"Generated {len(claims_df)} claims and {len(patients)} distinct geospatial patients!")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    claims_df.to_parquet(data_dir / "claims.parquet")
    patients.to_parquet(data_dir / "patients.parquet")
    providers.to_parquet(data_dir / "providers.parquet")
    pharmacies.to_parquet(data_dir / "pharmacies.parquet")
    
    return claims_df, patients, providers, pharmacies

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all()
