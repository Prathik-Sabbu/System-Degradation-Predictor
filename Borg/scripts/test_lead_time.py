import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import gc
import os

def main():
    # Attempt to locate data files
    base_dir = r"C:\Users\sabbu\OneDrive\Documents\Git\System-Degradation-Predictor"
    event_path = os.path.join(base_dir, r"Data\BORG\instance_events-000000000000.json.gz")
    usage_path = os.path.join(base_dir, r"Data\BORG\instance_usage-000000000000.json.gz")
    model_path = os.path.join(base_dir, r"Borg\models_Borg\fullscale_lgb_model_457.pkl")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, r"Borg\models\fullscale_lgb_model_457.pkl")

    print("1. Loading raw data with labels [4, 5, 7] and 8 Features...")
    event_schema = {
        "machine_id": pl.String, "time": pl.String, "type": pl.String,
        "priority": pl.String, "missing_type": pl.String,
        "resource_request": pl.Struct([pl.Field("cpus", pl.Float64), pl.Field("memory", pl.Float64)]),
        "constraint": pl.List(pl.Struct([])) 
    }
    usage_schema = {
        "machine_id": pl.String, "start_time": pl.String,
        "average_usage": pl.Struct([pl.Field("cpus", pl.Float64), pl.Field("memory", pl.Float64)]),
        "maximum_usage": pl.Struct([pl.Field("cpus", pl.Float64), pl.Field("memory", pl.Float64)]),
        "random_sample_usage": pl.Struct([pl.Field("cpus", pl.Float64)]),
        "assigned_memory": pl.Float64, "page_cache_memory": pl.Float64, 
        "cycles_per_instruction": pl.Float64, "memory_accesses_per_instruction": pl.Float64,
        "sample_rate": pl.Float64,
        "cpu_usage_distribution": pl.List(pl.Float64),
        "tail_cpu_usage_distribution": pl.List(pl.Float64)
    }

    failure_labels = [4, 5, 7]

    # Full dataset load for proper validation
    events = (pl.scan_ndjson(event_path, schema=event_schema)
              .select([
                  pl.col("machine_id").cast(pl.Float64), 
                  pl.col("time").cast(pl.Float64),
                  pl.col("type").cast(pl.Int32).is_in(failure_labels).cast(pl.Int8).alias("error_signal"),
                  pl.col("priority").cast(pl.Float64).alias("ev_priority"),
                  pl.col("missing_type").cast(pl.Float64).alias("ev_missing_type"),
                  pl.col("resource_request").struct.field("cpus").alias("ev_req_cpu"),
                  pl.col("resource_request").struct.field("memory").alias("ev_req_mem"),
                  pl.col("constraint").list.len().cast(pl.Float64).alias("ev_num_constraints")
              ])
              .sort("time"))

    usage = (pl.scan_ndjson(usage_path, schema=usage_schema)
             .select([
                 pl.col("machine_id").cast(pl.Float64), 
                 pl.col("start_time").cast(pl.Float64).alias("time"),
                 pl.col("sample_rate").alias("us_sample_rate"),
                 pl.col("page_cache_memory").alias("us_page_cache_mem"),
                 pl.col("memory_accesses_per_instruction").alias("us_mapi"),
                 pl.col("average_usage").struct.field("cpus").alias("us_avg_cpu"),
                 pl.col("average_usage").struct.field("memory").alias("us_avg_mem"),
                 pl.col("maximum_usage").struct.field("cpus").alias("us_max_cpu"),
                 pl.col("maximum_usage").struct.field("memory").alias("us_max_mem"),
                 pl.col("cpu_usage_distribution").list.last().cast(pl.Float64).alias("us_cpu_dist_tail"),
                 pl.col("random_sample_usage").struct.field("cpus").alias("us_rand_cpu"),
                 pl.col("cycles_per_instruction").alias("us_cpi"),
                 pl.col("assigned_memory").alias("us_assigned_mem"),
                 pl.col("tail_cpu_usage_distribution").list.last().cast(pl.Float64).alias("us_cpu_tail_dist_tail")
             ])
             .sort("time"))

    df_raw = usage.join_asof(events, on="time", by="machine_id", strategy="backward").collect().fill_null(0.0)

    print("2. Resampling to strict 5-minute time grid...")
    ev_features = [
        'ev_req_cpu', 
        'ev_priority', 
        'ev_num_constraints',
        'ev_req_mem'
    ]

    us_features = [
        'us_assigned_mem', 
        'us_cpi', 
        'us_page_cache_mem', 
        'us_avg_mem', 
        'us_avg_cpu', 
        'us_mapi', 
        'us_sample_rate',
        'us_cpu_dist_tail',
        'us_max_cpu'
    ]

    features = ev_features + us_features

    df_grid = df_raw.with_columns(
        (pl.col("time") // 300_000_000).cast(pl.Int64).alias("time_bucket")
    )

    df_grid = df_grid.group_by(["machine_id", "time_bucket"]).agg(
        [pl.col("error_signal").max()] + [pl.col(f).mean() for f in features]
    ).sort(["machine_id", "time_bucket"]).fill_null(0.0)

    print("3. Performing per-machine temporal split (70% train / 30% test)...")
    test_rows = []
    for _, group in df_grid.group_by("machine_id"):
        group = group.sort("time_bucket")
        cut = int(len(group) * 0.70)
        test_rows.append(group.slice(cut))
    df_test_grid = pl.concat(test_rows)

    # For accurate scaler we need train grid
    train_rows = []
    for _, group in df_grid.group_by("machine_id"):
        group = group.sort("time_bucket")
        cut = int(len(group) * 0.70)
        train_rows.append(group.slice(0, cut))
    df_train_grid = pl.concat(train_rows)

    del df_raw, df_grid
    gc.collect()

    print("4. Applying RobustScaler...")
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaler.fit(df_train_grid.select(features).to_numpy())
    df_test_scaled_vals = scaler.transform(df_test_grid.select(features).to_numpy())
    df_test_grid = df_test_grid.with_columns([
        pl.Series(name=features[i], values=df_test_scaled_vals[:, i]) for i in range(len(features))
    ])
    del df_train_grid
    gc.collect()

    print(f"5. Loading Model: {model_path}")
    loaded_lgb = joblib.load(model_path)
    
    print("6. Calculating Lead Times on Test Set...")
    window_size = 24
    MAX_GAP = 24
    
    lead_times = []
    
    # Threshold for predicting a failure
    THRESHOLD = 0.45

    genuine_misses = 0
    no_window_events = 0

    groups = df_test_grid.group_by("machine_id")
    
    for _, group in groups:
        buckets = group.select("time_bucket").to_numpy().flatten()
        if len(buckets) == 0: continue
            
        vals = group.select(features).to_numpy()
        signals = group.select("error_signal").to_numpy().flatten()
        
        chunks = []
        current_chunk_buckets = [buckets[0]]
        current_chunk_vals = [vals[0]]
        current_chunk_signals = [signals[0]]
        
        for i in range(1, len(buckets)):
            gap = buckets[i] - buckets[i-1]
            if gap > MAX_GAP:
                chunks.append((current_chunk_buckets, current_chunk_vals, current_chunk_signals))
                current_chunk_buckets = [buckets[i]]
                current_chunk_vals = [vals[i]]
                current_chunk_signals = [signals[i]]
            else:
                current_chunk_buckets.append(buckets[i])
                current_chunk_vals.append(vals[i])
                current_chunk_signals.append(signals[i])
                
        chunks.append((current_chunk_buckets, current_chunk_vals, current_chunk_signals))
        
        for c_buckets, c_vals, c_signals in chunks:
            if len(c_buckets) == 0: continue
            min_b = int(c_buckets[0])
            max_b = int(c_buckets[-1])
            length = max_b - min_b + 1
            if length < window_size + 6 or length > 100000: continue
                
            dense_vals = np.zeros((length, len(features)), dtype='float32')
            dense_signals = np.zeros(length, dtype='int8')
            
            current_val = c_vals[0]
            idx = 0
            for b in range(length):
                actual_bucket = min_b + b
                if idx < len(c_buckets) and c_buckets[idx] == actual_bucket:
                    current_val = c_vals[idx]
                    dense_signals[b] = c_signals[idx]
                    idx += 1
                dense_vals[b] = current_val
            
            # Predict on every possible window
            num_windows = length - window_size - 12
            if num_windows <= 0: continue
            
            X_windows = np.zeros((num_windows, window_size, len(features)), dtype='float32')
            for i in range(num_windows):
                X_windows[i] = dense_vals[i : i + window_size]
                
            # Apply engineer_tree_features
            last_step = X_windows[:, -1, :]
            mean_val = np.mean(X_windows, axis=1)
            delta_val = X_windows[:, -1, :] - X_windows[:, 0, :]
            std_val = np.std(X_windows, axis=1)
            max_val = np.max(X_windows, axis=1)
            
            X_flat = np.hstack([last_step, mean_val, delta_val, std_val, max_val])
            
            probs = loaded_lgb.predict_proba(X_flat)[:, 1]
            
            # Find actual failures (0 to 1 transitions, or just 1s)
            in_failure = False
            end_of_previous_failure = -1
            
            for i in range(num_windows):
                actual_fail = dense_signals[i + window_size : i + window_size + 12].sum() > 0
                if actual_fail and not in_failure:
                    in_failure = True
                    lookback_start = max(0, i - 288, end_of_previous_failure + 1)
                    earliest_pred = -1
                    any_eligible = False
                    for j in range(lookback_start, i):
                        if dense_signals[j: j + window_size].sum() > 0:
                            continue
                        any_eligible = True
                        if probs[j] > THRESHOLD:
                            earliest_pred = j
                            break

                    if earliest_pred != -1:
                        lead_times.append((i - earliest_pred) * 5)
                    elif any_eligible:
                        lead_times.append(0)          # genuine miss: had a chance, didn't fire
                        genuine_misses += 1
                    else:
                        no_window_events += 1         # no clean window existed at all -- not a model failure
                elif not actual_fail and in_failure:
                    in_failure = False
                    end_of_previous_failure = i + window_size + 12 - 1

    print(f"\n--- Lead Time Analysis ---")
    total_eval = len(lead_times) + no_window_events
    print(f"Total Failure Events Evaluated: {total_eval}")
    print(f"Events with no clean eligible window: {no_window_events} ({(no_window_events / total_eval * 100) if total_eval else 0:.1f}%)")
    
    if len(lead_times) > 0:
        lead_times = np.array(lead_times)
        caught = lead_times[lead_times > 0]
        print(f"\nOf the {len(lead_times)} events that had a clean warning window:")
        print(f"  Failures Caught Early: {len(caught)} ({len(caught)/len(lead_times)*100:.1f}%)")
        print(f"  Genuine Misses: {genuine_misses} ({genuine_misses/len(lead_times)*100:.1f}%)")
        print(f"  Average Lead Time (All eligible): {np.mean(lead_times):.1f} minutes")
        if len(caught) > 0:
            print(f"  Average Lead Time (When Caught): {np.mean(caught):.1f} minutes")
            print(f"  Max Lead Time: {np.max(caught)} minutes")
            print(f"  Min Lead Time: {np.min(caught)} minutes")

if __name__ == "__main__":
    main()
