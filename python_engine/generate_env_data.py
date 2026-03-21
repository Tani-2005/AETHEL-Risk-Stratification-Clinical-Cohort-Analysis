import pandas as pd
import numpy as np
import os

def generate_pan_european_data():
    print("Initializing AETHEL 100-City Environmental Engine...")
    
    # Load the 100-city metadata registry
    try:
        registry = pd.read_csv("data/raw/eu_registry.csv")
    except FileNotFoundError:
        print("Error: Run build_eu_registry.py first!")
        return
        
    dates = pd.date_range(start='2020-01-01', periods=60, freq='ME')
    all_data = []
    
    for _, row in registry.iterrows():
        np.random.seed(abs(hash(row['city_id'])) % (10**8)) 
        seasonality = np.cos(np.linspace(0, 10 * np.pi, 60)) * 5 
        
        data = {
            'date': dates,
            'location': row['city_id'],
            'pm2_5': np.maximum(row['base_pm25'] + seasonality + np.random.normal(0, 2, 60), 0),
            'no2': np.maximum(row['base_no2'] + (seasonality * 1.5) + np.random.normal(0, 3, 60), 0)
        }
        all_data.append(pd.DataFrame(data))
    
    df = pd.concat(all_data, ignore_index=True)
    output_path = "data/raw/regional_environmental_history.csv"
    df.to_csv(output_path, index=False)
    print(f"Success! Generated 60 months of history for {len(registry)} European hubs.")

if __name__ == "__main__":
    generate_pan_european_data()