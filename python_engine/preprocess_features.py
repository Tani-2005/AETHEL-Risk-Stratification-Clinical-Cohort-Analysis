import pandas as pd
import numpy as np
import os

def build_analytical_dataset():
    print("Aligning 100-City Telemetry with Clinical Cohort...")
    
    try:
        clinical_df = pd.read_csv("data/raw/synthetic_clinical_cohort.csv")
        env_df = pd.read_csv("data/raw/regional_environmental_history.csv")
        registry = pd.read_csv("data/raw/eu_registry.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Randomly assign patients to the 100 cities
    np.random.seed(42)
    clinical_df['city_id'] = np.random.choice(registry['city_id'], size=len(clinical_df))
    
    pm25_exposures, no2_exposures, lats, lons, city_names = [], [], [], [], []
    
    for _, patient in clinical_df.iterrows():
        months = max(1, int(patient['months_observed']))
        patient_city_id = patient['city_id']
        
        # Get city metadata
        city_meta = registry[registry['city_id'] == patient_city_id].iloc[0]
        city_env = env_df[env_df['location'] == patient_city_id].head(months)
        
        pm25_exposures.append(city_env['pm2_5'].mean())
        no2_exposures.append(city_env['no2'].mean())
        
        # Add slight map scatter so patients in the same city don't completely overlap
        lats.append(city_meta['lat'] + np.random.normal(0, 0.02))
        lons.append(city_meta['lon'] + np.random.normal(0, 0.02))
        city_names.append(city_meta['city'])
    
    clinical_df['avg_pm25_exposure'] = pm25_exposures
    clinical_df['avg_no2_exposure'] = no2_exposures
    clinical_df['lat'] = lats
    clinical_df['lon'] = lons
    clinical_df['city'] = city_names
    
    os.makedirs("data/processed", exist_ok=True)
    clinical_df.to_csv("data/processed/analytical_cohort.csv", index=False)
    print("Success! Engineered final analytical dataset across 100 European hubs.")

if __name__ == "__main__":
    build_analytical_dataset()