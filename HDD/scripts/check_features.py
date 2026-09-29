import polars as pl
import os
import glob

FEATURES = [
    'smart_197_raw', 'smart_187_raw', 'smart_9_raw', 'smart_5_raw',
    'smart_192_raw', 'smart_196_raw', 'smart_193_raw', 'smart_12_raw',
    'smart_3_raw', 'smart_7_raw', 'smart_194_raw', 'smart_241_raw',
    'smart_4_raw', 'smart_222_raw', 'smart_190_raw', 'smart_200_raw',
    'smart_2_raw', 'smart_199_raw', 'smart_188_raw', 'smart_195_raw',
    'smart_198_raw', 'smart_8_raw'
]

def check_features_in_folder(base_folder):
    search_path = os.path.join(base_folder, "**", "*.csv")
    csv_files = glob.glob(search_path, recursive=True)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("._") and "__MACOSX" not in f]
    
    if not csv_files:
        return
        
    # Grab the very first and last file to check the beginning and end of the year
    files_to_check = [csv_files[0], csv_files[-1]]
    
    print(f"\n=== Checking features in {base_folder} ===")
    for f in files_to_check:
        print(f"\nFile: {os.path.basename(f)}")
        # Read just 1 row to get the schema
        df = pl.read_csv(f, n_rows=1, ignore_errors=True)
        file_cols = set(df.columns)
        
        missing = [feat for feat in FEATURES if feat not in file_cols]
        present = [feat for feat in FEATURES if feat in file_cols]
        
        print(f"Features Present ({len(present)}/{len(FEATURES)}):")
        print("  " + ", ".join(present))
        
        if missing:
            print(f"Features MISSING ({len(missing)}/{len(FEATURES)}):")
            print("  " + ", ".join(missing))
        else:
            print("All required features are present!")

if __name__ == "__main__":
    check_features_in_folder(r"D:\HDD BlackBlaze Data\Q2023")
    check_features_in_folder(r"D:\HDD BlackBlaze Data\Q2024")
