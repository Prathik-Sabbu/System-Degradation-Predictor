import polars as pl
import glob
import os

FEATURES = [
    'smart_197_raw', 'smart_187_raw', 'smart_9_raw', 'smart_5_raw',
    'smart_192_raw', 'smart_196_raw', 'smart_193_raw', 'smart_12_raw',
    'smart_3_raw', 'smart_7_raw', 'smart_194_raw', 'smart_241_raw',
    'smart_4_raw', 'smart_222_raw', 'smart_190_raw', 'smart_200_raw',
    'smart_2_raw', 'smart_199_raw', 'smart_188_raw', 'smart_195_raw',
    'smart_198_raw', 'smart_8_raw'
]
COLS_TO_KEEP = ['date', 'serial_number', 'model', 'failure'] + FEATURES

def convert_to_parquet(base_folder, output_file):
    print(f"\n--- Starting conversion for {base_folder} ---")
    
    search_path = os.path.join(base_folder, "**", "*.csv")
    raw_files = glob.glob(search_path, recursive=True)
    
    # Filter out MacOS hidden files (._*) and __MACOSX folders that cause schema corruption
    csv_files = [f for f in raw_files if not os.path.basename(f).startswith("._") and "__MACOSX" not in f]
    
    if not csv_files:
        print(f"Error: No valid CSV files found in {base_folder}")
        return
        
    print(f"Found {len(csv_files)} valid CSV files. Unifying strict schema mapping...")
    
    try:
        lazy_dfs = []
        for f in csv_files:
            lf = pl.scan_csv(f, ignore_errors=True)
            file_cols = lf.collect_schema().names()
            
            exprs = []
            for c in COLS_TO_KEEP:
                if c in file_cols:
                    if c in FEATURES:
                        exprs.append(pl.col(c).cast(pl.Float32, strict=False))
                    elif c == 'failure':
                        exprs.append(pl.col(c).cast(pl.Int32, strict=False))
                    else:
                        exprs.append(pl.col(c).cast(pl.String, strict=False))
                else:
                    # Missing columns get injected as explicit nulls of the correct type
                    if c in FEATURES:
                        exprs.append(pl.lit(None).cast(pl.Float32).alias(c))
                    elif c == 'failure':
                        exprs.append(pl.lit(None).cast(pl.Int32).alias(c))
                    else:
                        exprs.append(pl.lit(None).cast(pl.String).alias(c))
            
            # Select ONLY our target columns, strictly typed
            lazy_dfs.append(lf.select(exprs))
            
        # Standard concat will now work perfectly since schemas are mathematically identical
        combined_lazy_df = pl.concat(lazy_dfs)
        
        print(f"Writing to {output_file} (This may take a few minutes...)")
        combined_lazy_df.sink_parquet(output_file)
        print(f"Success! Saved {output_file}")
    except Exception as e:
        print(f"Failed during processing: {e}")

if __name__ == "__main__":
    os.makedirs("Data", exist_ok=True)
    
    convert_to_parquet(
        base_folder=r"D:\HDD BlackBlaze Data\Q2026", 
        output_file="Data/HardDrive/Combined_HDD_Data_2026.parquet"
    )

