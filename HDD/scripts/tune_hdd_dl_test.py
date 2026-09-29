import gc
import os
import random
import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, precision_recall_curve

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
RAW_FEATURES = [
    'smart_197_raw', 'smart_187_raw', 'smart_9_raw', 'smart_5_raw',
    'smart_192_raw', 'smart_196_raw', 'smart_193_raw', 'smart_12_raw',
    'smart_3_raw', 'smart_7_raw', 'smart_194_raw', 'smart_241_raw',
    'smart_4_raw', 'smart_222_raw', 'smart_190_raw', 'smart_200_raw',
    'smart_2_raw', 'smart_199_raw', 'smart_188_raw', 'smart_195_raw',
    'smart_198_raw', 'smart_8_raw'
]

FEATURES = RAW_FEATURES + [f.replace('_raw', '_normalized') for f in RAW_FEATURES]

PARQUET_PATHS = [
    "Data/HardDrive/Combined_HDD_Data_2025.parquet",
    "Data/HardDrive/Combined_HDD_Data_2026.parquet",
]

LEAD_DAYS = 7
WINDOW_SIZES = [60]
CUTOFF = date(2026, 1, 1)
EPOCH = date(1970, 1, 1)

MAX_TRAIN_FAILS = 1000
MAX_TEST_FAILS = 300
SEED = 42

CONFIGS_TO_RUN = ["temporal_recent"]

# Deep Learning Hyperparams
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_int(d):
    return (d - EPOCH).days

def make_configs(data_end):
    last_t = data_end - timedelta(days=LEAD_DAYS)
    train_hi = CUTOFF - timedelta(days=LEAD_DAYS + 1)
    return {
        "control_random_2026": {
            "mode": "random",
            "lo": date(2026, 1, 1), "hi": last_t,
        },
        "temporal_recent": {
            "mode": "temporal",
            "train": (date(2025, 7, 1), train_hi),
            "test": (CUTOFF, last_t),
        },
    }

def get_serial_stats(parquet_paths):
    parts = []
    for p in parquet_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        lf = pl.scan_parquet(p).select([
            pl.col("serial_number").cast(pl.String),
            pl.col("date").cast(pl.String), 
            pl.col("failure").cast(pl.Int32, strict=False),
        ])
        parts.append(
            lf.group_by("serial_number").agg([
                pl.col("date").min().alias("min_date"),
                pl.col("date").max().alias("max_date"),
                pl.col("date").filter(pl.col("failure") == 1).max().alias("fail_date"),
            ]).collect()
        )
    
    stats = pl.concat(parts).group_by("serial_number").agg([
        pl.col("min_date").min(),
        pl.col("max_date").max(),
        pl.col("fail_date").max()
    ])
    
    return stats.with_columns([
        pl.col("min_date").str.to_date("%Y-%m-%d"),
        pl.col("max_date").str.to_date("%Y-%m-%d"),
        pl.col("fail_date").str.to_date("%Y-%m-%d")
    ])

def select_serials(stats, cfg, rng, train_ratio=4, test_ratio=50):
    def failed_pool(lo, hi):
        return stats.filter(
            pl.col("fail_date").is_not_null() &
            (pl.col("fail_date") >= lo) &
            (pl.col("fail_date") <= hi)
        )["serial_number"].to_list()

    def healthy_pool(lo, hi):
        return stats.filter(
            pl.col("fail_date").is_null() &
            (pl.col("max_date") >= hi) &
            (pl.col("min_date") <= lo)
        )["serial_number"].to_list()

    if cfg["mode"] == "random":
        bounds = {"train": (cfg["lo"], cfg["hi"]), "test": (cfg["lo"], cfg["hi"])}
        f_pool = failed_pool(cfg["lo"], cfg["hi"])
        rng.shuffle(f_pool)
        test_f = f_pool[:MAX_TEST_FAILS]
        train_f = f_pool[MAX_TEST_FAILS : MAX_TEST_FAILS + MAX_TRAIN_FAILS]
        h_pool = healthy_pool(cfg["lo"], cfg["hi"])
        rng.shuffle(h_pool)
        
        n_train_h = min(len(h_pool), train_ratio * len(train_f))
        train_h = h_pool[:n_train_h]
        
        # Test healthy gets bounded to test_ratio
        h_test_pool = h_pool[n_train_h:]
        n_test_h = min(len(h_test_pool), test_ratio * len(test_f))
        test_h = h_test_pool[:n_test_h]
    else:
        bounds = {"train": cfg["train"], "test": cfg["test"]}
        f_test_pool = failed_pool(*bounds["test"])
        f_train_pool = failed_pool(*bounds["train"])
        test_f = rng.sample(f_test_pool, min(MAX_TEST_FAILS, len(f_test_pool)))
        train_f = rng.sample(f_train_pool, min(MAX_TRAIN_FAILS, len(f_train_pool)))
        h_test_pool = healthy_pool(*bounds["test"])
        h_train_pool = healthy_pool(*bounds["train"])
        
        n_train_h = min(len(h_train_pool), train_ratio * len(train_f))
        train_h = rng.sample(h_train_pool, n_train_h)
        taken = set(train_h)
        
        # Test healthy gets bounded to test_ratio
        h_test_pool = [s for s in h_test_pool if s not in taken]
        n_test_h = min(len(h_test_pool), test_ratio * len(test_f))
        test_h = rng.sample(h_test_pool, n_test_h)

    return train_f, train_h, test_f, test_h, bounds

def load_histories(parquet_paths, serials):
    if not serials:
        return {}
        
    target_serials = set(serials)
    serial_series = pl.Series("serial_number", list(target_serials))
    
    lfs = []
    for p in parquet_paths:
        try:
            pf = pq.ParquetFile(p)
            schema_names = pf.schema.names
        except Exception:
            continue
            
        read_cols = ["serial_number", "date"] + [c for c in FEATURES if c in schema_names]
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i, columns=read_cols)
            df_chunk = pl.from_arrow(table)
            df_filtered = df_chunk.filter(pl.col("serial_number").is_in(serial_series))
            
            if df_filtered.height > 0:
                exprs = [
                    pl.col("serial_number").cast(pl.String),
                    pl.col("date").cast(pl.String).str.to_date("%Y-%m-%d").cast(pl.Int32).alias("day"),
                ]
                for c in FEATURES:
                    if c in schema_names:
                        exprs.append(pl.col(c).cast(pl.Float32, strict=False))
                    else:
                        exprs.append(pl.lit(None, dtype=pl.Float32).alias(c))
                lfs.append(df_filtered.select(exprs))

    if not lfs:
        return {}

    df = pl.concat(lfs).unique(subset=["serial_number", "day"], keep="first", maintain_order=True).sort(["serial_number", "day"])
    hist = {}
    for name, g in df.group_by("serial_number"):
        serial = name[0] if isinstance(name, tuple) else name
        hist[serial] = (g["day"].to_numpy(), g.select(FEATURES).to_numpy())
        
    del df, lfs, serial_series
    gc.collect()
    return hist

def extract_windows(hist, serials, fail_day_map, bounds, window, seed=SEED):
    lo_i, hi_i = to_int(bounds[0]), to_int(bounds[1])
    rng = np.random.default_rng(seed)
    X, y = [], []

    for s in serials:
        h = hist.get(s)
        if h is None:
            continue
        d, v = h
        n = len(d)
        if n < window:
            continue

        fail_date_int = fail_day_map.get(s)
        if fail_date_int is not None:
            target_t = fail_date_int - LEAD_DAYS
            if target_t > d[-1] or target_t < d[window - 1]:
                continue
            idx = np.searchsorted(d, target_t)
            if d[idx] == target_t:
                st = idx - window + 1
                if st >= 0 and d[idx] - d[st] == window - 1:
                    X.append(v[st:idx + 1])
                    y.append(1)
        else:
            valid_end_idxs = []
            for i in range(window - 1, n):
                if lo_i <= d[i] <= hi_i and (d[i] - d[i - window + 1] == window - 1):
                    if i + LEAD_DAYS < n and d[i + LEAD_DAYS] - d[i] == LEAD_DAYS:
                        valid_end_idxs.append(i)
            if valid_end_idxs:
                idx = rng.choice(valid_end_idxs)
                st = idx - window + 1
                X.append(v[st:idx + 1])
                y.append(0)

    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)

# --------------------------------------------------------------------------------------
# Deep Learning Architecture
# --------------------------------------------------------------------------------------
class DegradationNet(nn.Module):
    def __init__(self, num_features=44, hidden_size=64):
        super().__init__()
        # Batch Norm to handle raw unscaled SMART values
        self.bn1 = nn.BatchNorm1d(num_features)
        
        # 1D CNN to extract local temporal patterns (e.g., sudden spikes over 3 days)
        self.conv = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # LSTM to track long-term sequential degradation
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        
        # Classifier
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (Batch, Seq_len, Features)
        x = x.permute(0, 2, 1) # (Batch, Features, Seq_len) for Conv1D and BatchNorm
        x = self.bn1(x)
        x = self.relu(self.conv(x))
        
        x = x.permute(0, 2, 1) # (Batch, Seq_len, Features) for LSTM
        out, (hn, cn) = self.lstm(x)
        
        # Predict based on the final hidden state
        return torch.sigmoid(self.fc(hn[-1]))

def evaluate_dl(X_tr_3d, y_tr, X_te_3d, y_te):
    # Fill missing values with 0
    X_tr_3d = np.nan_to_num(X_tr_3d, nan=0.0)
    X_te_3d = np.nan_to_num(X_te_3d, nan=0.0)
    
    # Convert to tensors
    X_tr_t = torch.tensor(X_tr_3d, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_te_3d, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    # Rebalance batches mathematically using pos_weight
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = DegradationNet().to(DEVICE)
    criterion = nn.BCELoss() # Using BCELoss but manually injecting sample weights later if needed
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"      Training Neural Net on {DEVICE} for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Apply class weights manually to the loss
            weights = torch.where(labels == 1, pos_weight, 1.0)
            criterion.weight = weights
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        p = []
        # Predict in chunks to save memory
        for i in range(0, len(X_te_t), BATCH_SIZE):
            chunk = X_te_t[i:i+BATCH_SIZE].to(DEVICE)
            p.extend(model(chunk).cpu().numpy().flatten())
            
    p = np.array(p)
    pr_auc = average_precision_score(y_te, p)
    prec, rec, _ = precision_recall_curve(y_te, p)
    best_f1 = float(np.max((2 * prec * rec) / (prec + rec + 1e-9)))
    
    return pr_auc, best_f1

def main():
    print("Scanning datasets for per-drive stats...")
    stats = get_serial_stats(PARQUET_PATHS)
    data_end = stats["max_date"].max()
    print(f"  {len(stats):,} drives; data ends {data_end}")

    configs = make_configs(data_end)
    fail_days_all = {s: to_int(d) for s, d in zip(stats["serial_number"], stats["fail_date"]) if d is not None}

    summary = []
    for name in CONFIGS_TO_RUN:
        cfg = configs[name]
        print("\n" + "=" * 80)
        print(f"CONFIG: {name}")
        print("=" * 80)
        rng = random.Random(SEED)          
        
        train_f, train_h_max, test_f, test_h, bounds = select_serials(stats, cfg, rng, train_ratio=4)
        print(f"  Train window-end range: {bounds['train'][0]} -> {bounds['train'][1]}")
        print(f"  Test  window-end range: {bounds['test'][0]} -> {bounds['test'][1]}")
        
        train_serials_max = train_f + train_h_max
        test_serials = test_f + test_h
        fail_map = {s: fail_days_all[s] for s in (train_f + test_f)}
        
        print(f"  Loading TRAIN histories (max {len(train_serials_max)} drives)...")
        hist_train = load_histories(PARQUET_PATHS, train_serials_max)
        
        print(f"  Pre-computing TEST set tensors ({len(test_serials)} drives in chunks)...")
        test_data = {}
        X_te_chunks = {w: [] for w in WINDOW_SIZES}
        y_te_chunks = {w: [] for w in WINDOW_SIZES}
        
        chunk_sz = 20000
        for i in range(0, len(test_serials), chunk_sz):
            chunk = test_serials[i:i + chunk_sz]
            hist_test = load_histories(PARQUET_PATHS, chunk)
            for w in WINDOW_SIZES:
                X_3d, y = extract_windows(hist_test, chunk, fail_map, bounds["test"], w)
                if len(y) > 0:
                    X_te_chunks[w].append(X_3d)
                    y_te_chunks[w].append(y)
            del hist_test
            gc.collect()
            
        for w in WINDOW_SIZES:
            if y_te_chunks[w]:
                test_data[w] = (np.vstack(X_te_chunks[w]), np.concatenate(y_te_chunks[w]))
            else:
                test_data[w] = (np.array([]), np.array([]))
                
        del X_te_chunks, y_te_chunks
        gc.collect()

        # We will run both 3:1 and 4:1 for the optimal 60-day window
        for ratio in [3, 4]:
            print(f"\n  --- Testing Train Ratio: {ratio}:1 ---")
            current_train_h = train_h_max[:len(train_f) * ratio]
            train_serials = train_f + current_train_h
            
            for w in WINDOW_SIZES:
                X_tr_3d, y_tr = extract_windows(hist_train, train_serials, fail_map, bounds["train"], w)
                if len(y_tr) == 0 or y_tr.sum() == 0:
                    continue
                
                X_te_3d, y_te = test_data[w]
                if len(y_te) == 0 or y_te.sum() == 0:
                    continue
                
                base = y_te.mean()
                pr_auc, f1 = evaluate_dl(X_tr_3d, y_tr, X_te_3d, y_te)
                
                print(f"  window={w:>2} | train {len(y_tr):>5} ({int(y_tr.sum())} fail) | "
                      f"DL (PR:{pr_auc:.4f}, F1:{f1:.4f})")
                
                summary.append((name, ratio, w, int(y_tr.sum()), int(y_te.sum()), base, pr_auc, f1))
                gc.collect()

        del hist_train, test_data
        gc.collect()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'config':<22}{'ratio':>6}{'win':>4}{'tr_fail':>9}{'DL_PR':>9}{'DL_F1':>8}")
    for name, ratio, w, trf, tef, base, pa, f1 in summary:
        print(f"{name:<22}{ratio:>6}{w:>4}{trf:>9}{pa:>9.4f}{f1:>8.4f}")

if __name__ == "__main__":
    main()
