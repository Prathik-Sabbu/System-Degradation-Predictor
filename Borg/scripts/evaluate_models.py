import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import gc
import os

event_path = r"Data\BORG\instance_events-000000000000.json.gz"
usage_path = r"Data\BORG\instance_usage-000000000000.json.gz"

print("1. Loading raw data with labels [4, 5, 7] and 6 Features...")
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

print("Loading and Parsing Raw JSONs...")
failure_labels = [4, 5, 7]

events = (pl.scan_ndjson(event_path, n_rows=10000000, schema=event_schema)
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

usage = (pl.scan_ndjson(usage_path, n_rows=10000000, schema=usage_schema)
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

df_raw = events.join_asof(usage, on="time", by="machine_id", strategy="backward").collect().fill_null(0.0)

print("2. Resampling to strict 5-minute time grid...")
features = [
    'ev_req_cpu', 
    'ev_req_mem', 
    'us_sample_rate', 
    'us_page_cache_mem', 
    'us_avg_mem',
    'us_max_cpu',
    'us_avg_cpu',
    'us_cpi'
]

df_grid = df_raw.with_columns(
    (pl.col("time") // 300_000_000).cast(pl.Int64).alias("time_bucket")
)

df_grid = df_grid.group_by(["machine_id", "time_bucket"]).agg(
    [pl.col("error_signal").max()] + [pl.col(f).mean() for f in features]
).sort(["machine_id", "time_bucket"]).fill_null(0.0)

print("3. Performing per-machine temporal split (70% train / 30% test)...")
train_rows, test_rows = [], []
for _, group in df_grid.group_by("machine_id"):
    group = group.sort("time_bucket")
    cut = int(len(group) * 0.70)
    train_rows.append(group.slice(0, cut))
    test_rows.append(group.slice(cut))
df_train_grid = pl.concat(train_rows)
df_test_grid = pl.concat(test_rows)

del df_raw, df_grid
gc.collect()

print("4. Applying StandardScaler...")
scaler = StandardScaler()

df_train_scaled_vals = scaler.fit_transform(df_train_grid.select(features).to_numpy())
df_train_grid = df_train_grid.with_columns([
    pl.Series(name=features[i], values=df_train_scaled_vals[:, i]) for i in range(len(features))
])

df_test_scaled_vals = scaler.transform(df_test_grid.select(features).to_numpy())
df_test_grid = df_test_grid.with_columns([
    pl.Series(name=features[i], values=df_test_scaled_vals[:, i]) for i in range(len(features))
])

print("5. Extracting sequences (with contiguity gap checks)...")
def extract_sequences_from_df(df, window_size=60):
    groups = df.group_by("machine_id")
    X, y = [], []
    MAX_GAP = 24
    
    for _, group in groups:
        buckets = group.select("time_bucket").to_numpy().flatten()
        if len(buckets) == 0: 
            continue
            
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
            
            if max_b - min_b + 1 < window_size + 6:
                continue
                
            length = max_b - min_b + 1
            if length > 100000: 
                continue
                
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
                
            for T in range(window_size, length - 6):
                X.append(dense_vals[T - window_size : T])
                y.append(1 if dense_signals[T : T + 6].sum() > 0 else 0)
                
    return np.array(X), np.array(y)

window_size = 60

X_test, y_test = extract_sequences_from_df(df_test_grid, window_size)
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
print(f"Test Positive Rate: {y_test.mean():.4f}")
# Extract the last 5 timesteps for XGBoost and LightGBM models which were trained on 40 features
X_test_flat = X_test[:, -5:, :].reshape(X_test.shape[0], -1)

del df_train_grid, df_test_grid
gc.collect()

def evaluate_and_plot(y_true, y_prob, name):
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"--- {name} Performance ---")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    
    thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
    print(f"\n{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 48)
    for t in thresholds:
        y_pred_t = (y_prob > t).astype(int)
        p = precision_score(y_true, y_pred_t, zero_division=0)
        r = recall_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        print(f"{t:<10.2f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color='blue', label='Class 0 (Healthy)', density=True)
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color='red', label='Class 1 (Failure/Error)', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
    plt.xlabel('Predicted Probability (Confidence)')
    plt.ylabel('Density (Normalized Frequency)')
    plt.title(f'{name}: Histogram of Predicted Probabilities by True Class')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save directly to artifact directory
    artifact_dir = r"C:\Users\sabbu\.gemini\antigravity-ide\brain\1e719eb6-9b8f-4257-bb78-bff1a8b2a984"
    safe_name = name.split()[0].replace("[", "").replace("]", "")
    filename = os.path.join(artifact_dir, f"{safe_name}_histogram.png")
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

# # 1. Load and Evaluate XGBoost
# print("Loading XGBoost from disk...")
# loaded_xgb = xgb.XGBClassifier()
# loaded_xgb.load_model("models/fullscale_xgb_model_457.json")

# print("Generating predictions...")
# y_prob_xgb = loaded_xgb.predict_proba(X_test_flat)[:, 1]
# evaluate_and_plot(y_test, y_prob_xgb, "XGBoost [Labels 4, 5, 7]")

# 2. Load and Evaluate LightGBM
print("\nLoading LightGBM from disk...")
loaded_lgb = joblib.load('models/fullscale_lgb_model_457.pkl')

print("Generating predictions...")
y_prob_lgb = loaded_lgb.predict_proba(X_test_flat)[:, 1]
evaluate_and_plot(y_test, y_prob_lgb, "LightGBM [Labels 4, 5, 7]")

# # 3. Load and Evaluate CNN
# print("\nLoading CNN from disk...")
# loaded_cnn = load_model("models/Individual_CNN_model.keras")

# print("Generating predictions...")
# y_prob_cnn = loaded_cnn.predict(X_test).flatten()
# evaluate_and_plot(y_test, y_prob_cnn, "CNN [Labels 4, 5, 7]")

# # 4. Load and Evaluate GRU
# print("\nLoading GRU from disk...")
# loaded_gru = load_model("models/Individual_GRU_model.keras")

# print("Generating predictions...")
# y_prob_gru = loaded_gru.predict(X_test).flatten()
# evaluate_and_plot(y_test, y_prob_gru, "GRU [Labels 4, 5, 7]")

# # 5. Load and Evaluate LSTM
# print("\nLoading LSTM from disk...")
# loaded_lstm = load_model("models/Individual_LSTM_model.keras")

# print("Generating predictions...")
# y_prob_lstm = loaded_lstm.predict(X_test).flatten()
# evaluate_and_plot(y_test, y_prob_lstm, "LSTM [Labels 4, 5, 7]")

# # 6. Load and Evaluate TCN
# print("\nLoading TCN from disk...")
# # TCN might need custom_objects if it uses a custom layer, but we will try loading directly.
# loaded_tcn = load_model("models/TCN_model.keras")

# print("Generating predictions...")
# y_prob_tcn = loaded_tcn.predict(X_test).flatten()
# evaluate_and_plot(y_test, y_prob_tcn, "TCN [Labels 4, 5, 7]")

# # 7. Load and Evaluate 1d_ResNet
# print("\nLoading 1d_ResNet from disk...")
# loaded_tcn = load_model("models/1D_ResNet_model.keras")

# print("Generating predictions...")
# y_prob_tcn = loaded_tcn.predict(X_test).flatten()
# evaluate_and_plot(y_test, y_prob_tcn, "1D_ResNet [Labels 4, 5, 7]")

