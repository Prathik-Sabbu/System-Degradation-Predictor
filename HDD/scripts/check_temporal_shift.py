import polars as pl
import os

FEATURES = [
    'smart_197_raw', 'smart_187_raw', 'smart_9_raw', 'smart_5_raw',
    'smart_192_raw', 'smart_196_raw', 'smart_193_raw', 'smart_12_raw',
    'smart_3_raw', 'smart_7_raw', 'smart_194_raw', 'smart_241_raw',
    'smart_4_raw', 'smart_222_raw', 'smart_190_raw', 'smart_200_raw',
    'smart_2_raw', 'smart_199_raw', 'smart_188_raw', 'smart_195_raw',
    'smart_198_raw', 'smart_8_raw'
]

def run_diagnostics():
    parquet_paths = [
        "Data/HardDrive/Combined_HDD_Data_2023.parquet", 
        "Data/HardDrive/Combined_HDD_Data_2024.parquet",
        "Data/HardDrive/Combined_HDD_Data_2025.parquet" 
    ]
    
    print("=== DIAGNOSTIC 1: Null Rate Check by Year ===")
    for p in parquet_paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
            
        lf = pl.scan_parquet(p)
        cols = lf.collect_schema().names()
        
        features_to_check = [c for c in FEATURES if c in cols]
        
        print(f"\nScanning: {os.path.basename(p)}")
        
        # Calculate mean of is_null() for each feature
        null_rates = lf.select([pl.col(c).is_null().mean().alias(c) for c in features_to_check]).collect()
        
        # Format the output
        null_dict = null_rates.to_dicts()[0]
        for f, rate in null_dict.items():
            print(f"  {f}: {rate*100:.2f}% null")

    print("\n=== DIAGNOSTIC 2: Fleet Composition Shift (Model Overlap) ===")
    all_data = []
    for p in parquet_paths:
        if os.path.exists(p):
            # Scan only date and model columns
            all_data.append(pl.scan_parquet(p).select(['date', 'model']))
            
    if all_data:
        combined = pl.concat(all_data, how="diagonal")
        
        print("\nAggregating Top 15 Models for Train Period (Pre-July 2025)...")
        train_models = (
            combined.filter(pl.col('date') < '2025-07-01')
            .group_by('model')
            .agg(pl.col('model').count().alias('count'))
            .sort('count', descending=True)
            .limit(15)
            .collect()
        )
        
        print("Aggregating Top 15 Models for Test Period (Post-July 2025)...")
        test_models = (
            combined.filter(pl.col('date') >= '2025-07-01')
            .group_by('model')
            .agg(pl.col('model').count().alias('count'))
            .sort('count', descending=True)
            .limit(15)
            .collect()
        )
        
        print("\nTop 15 Models (Pre-July 2025):")
        print(train_models)
        
        print("\nTop 15 Models (Post-July 2025):")
        print(test_models)

if __name__ == "__main__":
    run_diagnostics()
