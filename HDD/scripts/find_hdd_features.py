import polars as pl
import xgboost as xgb
import numpy as np
import glob
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

def find_best_features():
    q4_path = "Data/HardDrive/data_Q4_2025/data_Q4_2025/"
    csv_files = glob.glob(q4_path + "*.csv")
    csv_files.sort()
    
    LEAD_DAYS = 7
    
    print(f"1. Pass 1: Finding all failure dates across {len(csv_files)} files...")
    failed_drives = {} 
    for file in csv_files:
        df = pl.read_csv(file, columns=['date', 'serial_number', 'failure'])
        fails = df.filter(pl.col('failure') == 1)
        if len(fails) > 0:
            for row in fails.iter_rows(named=True):
                failed_drives[row['serial_number']] = datetime.strptime(row['date'], '%Y-%m-%d')
                
    print(f"   Found {len(failed_drives)} total failures in Q4.")
    
    target_dates_range_for_failed = {
        serial: [
            (date - timedelta(days=LEAD_DAYS+1)).strftime('%Y-%m-%d'),
            (date - timedelta(days=LEAD_DAYS)).strftime('%Y-%m-%d'),
            (date - timedelta(days=LEAD_DAYS-1)).strftime('%Y-%m-%d')
        ]
        for serial, date in failed_drives.items()
    }
    
    print(f"2. Pass 2: Extracting data ~{LEAD_DAYS} days prior to failure, and deduplicating healthy samples...")
    
    sample_df = pl.read_csv(csv_files[0], n_rows=1)
    smart_columns = [col for col in sample_df.columns if col.startswith('smart_') and col.endswith('_raw')]
    
    # Drop known model-sparsity artifacts (like smart_18_raw which is 100% missing for Toshiba/WDC)
    artifacts_to_drop = ['smart_18_raw']
    smart_columns = [c for c in smart_columns if c not in artifacts_to_drop]
    
    columns_to_load = ['date', 'serial_number', 'model', 'failure'] + smart_columns
    
    schema_dict = {'date': pl.String, 'serial_number': pl.String, 'model': pl.String, 'failure': pl.Int32}
    for col in smart_columns:
        schema_dict[col] = pl.Float64
        
    sampled_healthy_serials = set()
    found_precursors = set() 
    
    all_target_failures = []
    all_healthy_samples = []
    failed_serials_list = list(failed_drives.keys())
    
    for file in csv_files:
        df = pl.read_csv(file, schema_overrides=schema_dict, ignore_errors=True)
        if len(df) == 0:
            continue
            
        for col in columns_to_load:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(schema_dict[col]).alias(col))
        df = df.select(columns_to_load)
            
        file_date = df['date'][0] 
        
        serials_needed_today = [
            s for s, dates in target_dates_range_for_failed.items() 
            if file_date in dates and s not in found_precursors
        ]
        
        if serials_needed_today:
            todays_targets = df.filter(pl.col('serial_number').is_in(serials_needed_today))
            if len(todays_targets) > 0:
                todays_targets = todays_targets.with_columns(pl.lit(1).alias('failure'))
                all_target_failures.append(todays_targets)
                found_precursors.update(todays_targets['serial_number'].to_list())
            
        healthy = df.filter(~pl.col('serial_number').is_in(failed_serials_list))
        if sampled_healthy_serials:
            healthy = healthy.filter(~pl.col('serial_number').is_in(list(sampled_healthy_serials)))
            
        if len(healthy) > 0:
            sample_size = min(len(healthy), 500)
            sampled = healthy.sample(n=sample_size, seed=42)
            all_healthy_samples.append(sampled)
            sampled_healthy_serials.update(sampled['serial_number'].to_list())
            
    failures_df = pl.concat(all_target_failures) if all_target_failures else pl.DataFrame(schema=df.schema)
    healthy_df = pl.concat(all_healthy_samples) if all_healthy_samples else pl.DataFrame(schema=df.schema)
    
    match_rate = (len(failures_df) / len(failed_drives)) * 100 if len(failed_drives) > 0 else 0
    print(f"   Extracted {len(failures_df)} valid failure-precursors (Match Rate: {match_rate:.1f}%) and {len(healthy_df)} healthy samples.")
    
    if len(failures_df) == 0:
        print("ERROR: No failure precursors found!")
        return
        
    target_healthy = len(failures_df) * 3
    if len(healthy_df) > target_healthy:
        healthy_df = healthy_df.sample(n=target_healthy, seed=42)
        
    final_df = pl.concat([failures_df, healthy_df])
    
    print("3. Handling Sparsity and 5-Fold Cross-Validation...")
    X = final_df.select([pl.col(c).cast(pl.Float32) for c in smart_columns]).to_numpy()
    y = final_df.select('failure').to_numpy().flatten()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    f1_scores = []
    precisions = []
    recalls = []
    
    print(f"   Running 5-Fold CV on {len(y)} total samples...")
    
    m_final = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        m = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        f1_scores.append(f1_score(y_test, preds))
        precisions.append(precision_score(y_test, preds))
        recalls.append(recall_score(y_test, preds))
        
    print(f"\n   --- 7-DAY FORECAST DIAGNOSTICS (5-Fold CV Average) ---")
    print(f"   F1 Score:  {np.mean(f1_scores):.4f}  (± {np.std(f1_scores):.4f})")
    print(f"   Precision: {np.mean(precisions):.4f}  (± {np.std(precisions):.4f})")
    print(f"   Recall:    {np.mean(recalls):.4f}  (± {np.std(recalls):.4f})")
    
    m_final.fit(X, y)
    importances = m_final.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\n=================================================")
    print(f" TOP 15 S.M.A.R.T. FEATURES ({LEAD_DAYS}-DAY FORECAST) ")
    print("=================================================")
    for i in range(15):
        idx = sorted_idx[i]
        feat = smart_columns[idx]
        imp = importances[idx] * 100
        if imp > 0:
            print(f"{i+1}. {feat.ljust(20)}: {imp:.2f}% importance")

if __name__ == "__main__":
    find_best_features()
