import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import gc
import os

event_path = r"Data\BORG\instance_events-*.json.gz"
usage_path = r"Data\BORG\instance_usage-*.json.gz"

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

print("1. Loading raw data...")
failure_labels = [4, 5, 7]

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

ev_features = ['ev_req_cpu', 'ev_req_mem', 'ev_priority', 'ev_missing_type', 'ev_num_constraints']
us_features = ['us_sample_rate', 'us_page_cache_mem', 'us_avg_mem', 'us_max_cpu', 
               'us_avg_cpu', 'us_cpi', 'us_cpu_dist_tail', 'us_rand_cpu', 'us_mapi', 'us_assigned_mem']
features = ev_features + us_features

print("2. Bucketing Usage and Events separately, then joining...")
usage_grid = usage.with_columns((pl.col("time") // 300_000_000).cast(pl.Int64).alias("time_bucket")).group_by(["machine_id", "time_bucket"]).agg([pl.col(f).mean() for f in us_features]).collect()
events_grid = events.with_columns((pl.col("time") // 300_000_000).cast(pl.Int64).alias("time_bucket")).group_by(["machine_id", "time_bucket"]).agg([pl.col("error_signal").max()] + [pl.col(f).mean() for f in ev_features]).collect()

df_grid = usage_grid.join(events_grid, on=["machine_id", "time_bucket"], how="left").sort(["machine_id", "time_bucket"])
df_grid = df_grid.with_columns(pl.col("error_signal").fill_null(0)).fill_null(0.0)

del usage_grid, events_grid
gc.collect()

print("3. Performing per-machine temporal split (Train only for feature importance)...")
train_rows = []
for _, group in df_grid.group_by("machine_id"):
    group = group.sort("time_bucket")
    cut = int(len(group) * 0.70)
    train_rows.append(group.slice(0, cut))
df_train_grid = pl.concat(train_rows)
del df_grid
gc.collect()

print("4. Scaling...")
scaler = RobustScaler()
df_train_scaled_vals = scaler.fit_transform(df_train_grid.select(features).to_numpy())
df_train_grid = df_train_grid.with_columns([
    pl.Series(name=features[i], values=df_train_scaled_vals[:, i]) for i in range(len(features))
])

print("5. Extracting sequences (with leak checks)...")
window_size = 24
X, y = [], []
groups = df_train_grid.group_by("machine_id")

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
        if buckets[i] - buckets[i-1] > 24:
            chunks.append((current_chunk_buckets, current_chunk_vals, current_chunk_signals))
            current_chunk_buckets, current_chunk_vals, current_chunk_signals = [buckets[i]], [vals[i]], [signals[i]]
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
        if length < window_size + 12 or length > 100000: continue
            
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
            
        for T in range(window_size, length - 12):
            if dense_signals[T - window_size : T].sum() == 0:
                X.append(dense_vals[T - window_size : T])
                y.append(1 if dense_signals[T : T + 12].sum() > 0 else 0)

X_train = np.array(X)
y_train = np.array(y)
print(f"Extracted Train Shapes: X={X_train.shape}, y={y_train.shape}")

print("6. Balancing Data and Engineering Tree Features...")
def undersample_to_ratio(X, y, target_healthy_ratio=0.50):
    fail_idx = np.where(y == 1)[0]
    healthy_idx = np.where(y == 0)[0]
    target_failure_ratio = 1.0 - target_healthy_ratio
    multiplier = target_healthy_ratio / target_failure_ratio
    target_healthy_count = int(len(fail_idx) * multiplier)
    target_healthy_count = min(target_healthy_count, len(healthy_idx))
    rng = np.random.default_rng(42)
    sampled_healthy_idx = rng.choice(healthy_idx, size=target_healthy_count, replace=False)
    keep_idx = np.concatenate([fail_idx, sampled_healthy_idx])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]

X_train_balanced, y_train_balanced = undersample_to_ratio(X_train, y_train, target_healthy_ratio=0.50)

def engineer_tree_features(X_3d):
    mean_vals = np.mean(X_3d, axis=1)
    std_vals = np.std(X_3d, axis=1)
    max_vals = np.max(X_3d, axis=1)
    min_vals = np.min(X_3d, axis=1)
    trend_vals = X_3d[:, -1, :] - X_3d[:, 0, :]
    last_vals = X_3d[:, -1, :]
    return np.hstack([mean_vals, std_vals, max_vals, min_vals, trend_vals, last_vals])

X_train_flat = engineer_tree_features(X_train_balanced)

# Create feature names
engineered_feature_names = []
for stat in ['mean', 'std', 'max', 'min', 'trend', 'last']:
    for f in features:
        engineered_feature_names.append(f"{f}_{stat}")

print("7. Training XGBoost to get pure importances...")
m = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
m.fit(X_train_flat, y_train_balanced)

importances = m.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n--- Top 20 Most Important Engineered Features ---")
with open("true_feature_importances.txt", "w") as f:
    for i in range(len(engineered_feature_names)):
        idx = sorted_idx[i]
        feat = engineered_feature_names[idx]
        imp = importances[idx] * 100
        line = f"{feat}: {imp:.2f}%"
        if i < 20:
            print(line)
        f.write(line + "\n")

# Aggregate by base feature
base_importances = {f: 0.0 for f in features}
for i, name in enumerate(engineered_feature_names):
    # Strip the suffix (_mean, _std, etc) to get the base feature name
    base_name = "_".join(name.split("_")[:-1]) 
    if base_name in base_importances:
        base_importances[base_name] += importances[i]

sorted_base = sorted(base_importances.items(), key=lambda item: item[1], reverse=True)

print("\n--- Base Feature Importance (Summed across Mean/Std/Trend etc) ---")
with open("true_base_feature_importances.txt", "w") as f:
    for feat, imp in sorted_base:
        val = imp * 100
        line = f"{feat}: {val:.2f}%"
        print(line)
        f.write(line + "\n")

print("\nSaved full rankings to true_feature_importances.txt and true_base_feature_importances.txt")
