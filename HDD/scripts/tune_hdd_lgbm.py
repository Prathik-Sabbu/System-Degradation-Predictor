import os
import time
import numpy as np
import polars as pl
import random
import lightgbm as lgb
import joblib
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve

FEATURES = list(set([
    'smart_200_raw', 'smart_197_raw', 'smart_187_raw', 'smart_5_raw', 
    'smart_196_raw', 'smart_2_raw', 'smart_199_raw', 'smart_12_raw', 
    'smart_9_raw', 'smart_188_raw', 'smart_195_raw', 'smart_190_raw', 
    'smart_241_raw', 'smart_198_raw', 'smart_8_raw',
    'smart_192_raw', 'smart_193_raw', 'smart_3_raw', 'smart_7_raw', 
    'smart_194_raw', 'smart_4_raw', 'smart_222_raw'
]))


def extract_hdd_sequences(parquet_path, window_size=14, lead_days=7):
    lazy_df = pl.scan_parquet(parquet_path)
    
    fails_df = lazy_df.filter(pl.col('failure') == 1).select(['serial_number']).collect()
    failed_serials = fails_df['serial_number'].unique().to_list()
    failed_set = set(failed_serials)
    
    all_serials = lazy_df.select('serial_number').unique().collect()['serial_number'].to_list()
    healthy_serials = [s for s in all_serials if s not in failed_set]
    
    rng = random.Random(42)
    target_healthy = len(failed_serials) * 3
    if len(healthy_serials) > target_healthy:
        sampled_healthy = set(rng.sample(healthy_serials, target_healthy))
    else:
        sampled_healthy = set(healthy_serials)
        
    serials_to_keep = list(failed_set.union(sampled_healthy))
    
    columns_to_keep = ['date', 'serial_number', 'failure'] + FEATURES
    df = (
        lazy_df
        .filter(pl.col('serial_number').is_in(serials_to_keep))
        .select(columns_to_keep)
        .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURES])
        .with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
        .sort(['serial_number', 'date'])
        .collect()
    )
    
    X, y = [], []
    groups = df.group_by('serial_number')
    
    for name, group in groups:
        serial = name[0]
            
        dates = group['date'].to_list()
        vals = group[FEATURES].to_numpy()
        
        if len(dates) < window_size:
            continue
            
        if serial in failed_set:
            fail_date = dates[-1] 
            target_date = fail_date - timedelta(days=lead_days)
            start_date = target_date - timedelta(days=window_size - 1)
            try:
                target_idx = dates.index(target_date)
            except ValueError:
                continue
            start_idx = target_idx - window_size + 1
            if start_idx < 0 or (dates[target_idx] - dates[start_idx]).days != window_size - 1:
                continue
            X.append(vals[start_idx : target_idx + 1])
            y.append(1)
        else:
            valid_starts = []
            for i in range(len(dates) - window_size + 1):
                if (dates[i + window_size - 1] - dates[i]).days == window_size - 1:
                    valid_starts.append(i)
            if valid_starts:
                start_idx = rng.choice(valid_starts)
                X.append(vals[start_idx : start_idx + window_size])
                y.append(0)
                
    return np.array(X), np.array(y)

def engineer_tree_features(X_3d):
    # Notice we use nanmean and nanmax here so LightGBM can still naturally handle NaNs!
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        last_step = X_3d[:, -1, :]
        mean_val = np.nanmean(X_3d, axis=1)
        delta_val = X_3d[:, -1, :] - X_3d[:, 0, :]
        max_val = np.nanmax(X_3d, axis=1)
        
    return np.hstack([last_step, mean_val, delta_val, max_val])

def find_optimal_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def main():
    parquet_path = "Data/HardDrive/Combined_HDD_Data_2025.parquet"
    if not os.path.exists(parquet_path):
        print("Missing parquet file.")
        return
        
    # Fixed forecast target
    LEAD_DAYS = 7 
    
    # We will test how many days of history LightGBM actually needs
    WINDOW_SIZES_TO_TEST = [14, 28] 
    
    results = {}
    best_f1 = 0
    best_window = None
    best_model = None
    best_threshold = 0.5
    
    print(f"=== OPTIMIZING LIGHTGBM (Target: Forecast {LEAD_DAYS} days in advance) ===")
    
    for w_size in WINDOW_SIZES_TO_TEST:
        print(f"\nEvaluating Window Size: {w_size} days...")
        X, y = extract_hdd_sequences(parquet_path, window_size=w_size, lead_days=LEAD_DAYS)
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        fail_idx = np.where(y_train == 1)[0]
        healthy_idx = np.where(y_train == 0)[0]
        sampled_healthy_idx = np.random.default_rng(42).choice(healthy_idx, size=len(fail_idx), replace=False)
        
        keep_idx = np.concatenate([fail_idx, sampled_healthy_idx])
        np.random.default_rng(42).shuffle(keep_idx)
        
        X_train_balanced = X_train_raw[keep_idx]
        y_train_balanced = y_train[keep_idx]
        
        # We don't impute/scale for LightGBM! Let it use native NaN handling
        X_train_tree = engineer_tree_features(X_train_balanced)
        X_test_tree = engineer_tree_features(X_test_raw)
        
        # Train
        m_lgb = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=200)
        m_lgb.fit(X_train_tree, y_train_balanced)
        
        # Predict probabilities
        y_probs = m_lgb.predict_proba(X_test_tree)[:, 1]
        
        # Show trade-offs across common thresholds
        print("\n  --- Threshold Trade-offs ---")
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            y_preds_t = (y_probs >= thresh).astype(int)
            p = precision_score(y_test, y_preds_t)
            r = recall_score(y_test, y_preds_t)
            f = f1_score(y_test, y_preds_t)
            print(f"  Threshold {thresh:.2f} -> F1: {f:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")
            
        # Find optimal threshold
        opt_thresh, opt_f1 = find_optimal_threshold(y_test, y_probs)
        
        # Get final binary predictions using that threshold
        y_preds_opt = (y_probs >= opt_thresh).astype(int)
        precision = precision_score(y_test, y_preds_opt)
        recall = recall_score(y_test, y_preds_opt)
        
        print(f"\n  -> Math-Optimal Threshold (Max F1): {opt_thresh:.3f}")
        print(f"  -> F1 Score: {opt_f1:.4f}  (Precision: {precision:.3f}, Recall: {recall:.3f})")
        
        results[w_size] = {
            "f1": opt_f1,
            "threshold": opt_thresh,
            "precision": precision,
            "recall": recall
        }
        
        if opt_f1 > best_f1:
            best_f1 = opt_f1
            best_window = w_size
            best_model = m_lgb
            best_threshold = opt_thresh
            
    print("FINAL RECOMMENDATION")
    print(f"Optimal Window Size : {best_window} days")
    print(f"Optimal Threshold   : {best_threshold:.3f}")
    print(f"Highest F1 Score    : {best_f1:.4f}")
    
    # Save the absolute best model
    model_filename = f"lgbm_hdd_lead{LEAD_DAYS}_window{best_window}.pkl"
    joblib.dump(best_model, model_filename)
    print(f"\nSaved best LightGBM model to '{model_filename}'")
    
if __name__ == "__main__":
    main()
