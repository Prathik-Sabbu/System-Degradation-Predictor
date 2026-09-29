import gc
import os
import random
import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve

warnings.filterwarnings("ignore", message="X does not have valid feature names")

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
    "Data/HardDrive/Combined_HDD_Data_2023.parquet", 
    "Data/HardDrive/Combined_HDD_Data_2024.parquet",
    "Data/HardDrive/Combined_HDD_Data_2025.parquet",      # 2025
    "Data/HardDrive/Combined_HDD_Data_2026.parquet", # 2026
]

LEAD_DAYS = 7
WINDOW_SIZES = [14, 28, 45, 60]
CUTOFF = date(2026, 1, 1)
EPOCH = date(1970, 1, 1)

MAX_TRAIN_FAILS = 500000  # Massive limit to capture all possible failures in the bounds
MAX_TEST_FAILS = 500000

#TEST_HEALTHY_RATIO = 325     # ~true base rate for a 3-month quarter (1.24% AFR / 4)
SEED = 42

# Which experiments to run (names come from make_configs)
CONFIGS_TO_RUN = ["temporal_full_decay"]


def to_int(d):
    """date -> integer days since epoch (matches polars Date -> Int32)."""
    return (d - EPOCH).days


def make_configs(data_end):
    last_t = data_end - timedelta(days=LEAD_DAYS)
    train_hi = CUTOFF - timedelta(days=LEAD_DAYS + 1)
    return {
        "temporal_full_decay": {
            "mode": "temporal",
            "train": (date(2023, 1, 1), train_hi),  # Pulling from Jan 2023 through Dec 2025
            "test": (CUTOFF, last_t),               # Testing strictly on 2026 unseen future
        },
    }


# --------------------------------------------------------------------------------------
# Per-drive stats and serial selection
# --------------------------------------------------------------------------------------
def _collect(lf):
    try:
        return lf.collect(engine="streaming")
    except TypeError:                       # older polars
        return lf.collect(streaming=True)


def get_serial_stats(parquet_paths):
    """One row per drive: first/last record date and the date of its failure=1 row (if any)."""
    parts = []
    for p in parquet_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        print(f"  Aggregating {p} ...")
        # FATAL OOM FIX: Do NOT parse .str.to_date() here! Doing string-to-date across 350M rows 
        # blows up Kaggle's RAM. String comparison YYYY-MM-DD works perfectly for min/max.
        lf = pl.scan_parquet(p).select([
            pl.col("serial_number").cast(pl.String),
            pl.col("date").cast(pl.String), 
            pl.col("failure").cast(pl.Int32, strict=False),
        ])
        parts.append(_collect(
            lf.group_by("serial_number").agg([
                pl.col("date").min().alias("min_date"),
                pl.col("date").max().alias("max_date"),
                pl.col("date").filter(pl.col("failure") == 1).max().alias("fail_date"),
            ])
        ))
    
    # We delay the .str.to_date() until after the data is reduced to just ~120,000 drives!
    return pl.concat(parts).group_by("serial_number").agg([
        pl.col("min_date").min().str.to_date("%Y-%m-%d"),
        pl.col("max_date").max().str.to_date("%Y-%m-%d"),
        pl.col("fail_date").max().str.to_date("%Y-%m-%d"),
    ])


def select_serials(stats, cfg, rng, train_ratio=4, test_ratio=50):
    """
    Returns (train_failed, train_healthy, test_failed, test_healthy, bounds)
    where bounds = {'train': (lo, hi), 'test': (lo, hi)} are inclusive limits on the
    window END date T for each role.
    """
    failed = stats.filter(pl.col("fail_date").is_not_null()).with_columns(
        pl.col("fail_date").dt.offset_by(f"-{LEAD_DAYS}d").alias("t_fail")
    )
    healthy = stats.filter(pl.col("fail_date").is_null())

    def failed_pool(lo, hi):
        return failed.filter((pl.col("t_fail") >= lo) & (pl.col("t_fail") <= hi))["serial_number"].to_list()

    def healthy_pool(lo, hi):
        # loose eligibility: drive has records overlapping [lo, hi]; exact validity is
        # enforced per-window during extraction
        return healthy.filter(
            (pl.col("min_date") <= hi)
            & (pl.col("max_date") >= lo + timedelta(days=LEAD_DAYS))
        )["serial_number"].to_list()

    if cfg["mode"] == "random":
        lo, hi = cfg["lo"], cfg["hi"]
        bounds = {"train": (lo, hi), "test": (lo, hi)}
        f_pool = failed_pool(lo, hi)
        rng.shuffle(f_pool)
        test_f = f_pool[:MAX_TEST_FAILS]
        train_f = f_pool[MAX_TEST_FAILS:MAX_TEST_FAILS + MAX_TRAIN_FAILS]
        h_test_pool = h_train_pool = healthy_pool(lo, hi)
    else:
        bounds = {"train": cfg["train"], "test": cfg["test"]}
        f_test_pool = failed_pool(*bounds["test"])
        f_train_pool = failed_pool(*bounds["train"])
        test_f = rng.sample(f_test_pool, min(MAX_TEST_FAILS, len(f_test_pool)))
        train_f = rng.sample(f_train_pool, min(MAX_TRAIN_FAILS, len(f_train_pool)))
        h_test_pool = healthy_pool(*bounds["test"])
        h_train_pool = healthy_pool(*bounds["train"])

    # Reserve train healthy first
    if train_ratio is None:
        n_train_h = len(h_train_pool)
    else:
        n_train_h = min(len(h_train_pool), train_ratio * len(train_f))
    train_h = rng.sample(h_train_pool, n_train_h)
    
    # Test healthy
    taken = set(train_h)
    h_test_pool = [s for s in h_test_pool if s not in taken]
    
    if test_ratio is None:
        n_test_h = len(h_test_pool)
    else:
        n_test_h = min(len(h_test_pool), test_ratio * len(test_f))
    test_h = rng.sample(h_test_pool, n_test_h)

    return train_f, train_h, test_f, test_h, bounds


# --------------------------------------------------------------------------------------
# Loading and window extraction
# --------------------------------------------------------------------------------------
def load_histories(parquet_paths, serials):
    """serial -> (days_int32[n], values_float32[n, n_features]) sorted by day."""
    if not serials:
        return {}
        
    target_serials = set(serials)
    serial_series = pl.Series(list(target_serials))
    
    lfs = []
    for p in parquet_paths:
        try:
            pf = pq.ParquetFile(p)
            schema_names = pf.schema.names
        except Exception:
            continue
            
        read_cols = ["serial_number", "date"] + [c for c in FEATURES if c in schema_names]
        
        # Read the file chunk-by-chunk (row group by row group)
        # This mathematically guarantees RAM never exceeds the size of one row group (~200MB)
        # and completely avoids Polars lazy engine CPU lockups.
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i, columns=read_cols)
            df_chunk = pl.from_arrow(table)
            
            # Filter chunk immediately using pre-compiled Polars Series for speed
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

    df = pl.concat(lfs).unique(subset=["serial_number", "day"], keep="first", maintain_order=True).sort(["serial_number", "day"])

    hist = {}
    for name, g in df.group_by("serial_number"):
        serial = name[0] if isinstance(name, tuple) else name
        hist[serial] = (g["day"].to_numpy(), g.select(FEATURES).to_numpy())
        
    del df, lfs, serial_list
    gc.collect()
    return hist


def extract_windows(hist, serials, fail_day_map, bounds, window, seed=SEED):
    lo_i, hi_i = to_int(bounds[0]), to_int(bounds[1])
    rng = np.random.default_rng(seed)
    X, y, end_days = [], [], []

    for s in serials:
        h = hist.get(s)
        if h is None: continue
        d, v = h
        n = len(d)
        if n < window: continue

        if s in fail_day_map:
            t = fail_day_map[s] - LEAD_DAYS
            if t < lo_i or t > hi_i: continue
            e = int(np.searchsorted(d, t))
            if e >= n or d[e] != t: continue
            st = e - window + 1
            if st < 0 or (d[e] - d[st]) != window - 1: continue
            
            X.append(v[st:e + 1])
            y.append(1)
            end_days.append(d[e]) # Capture exact day for decay calculation
        else:
            ends = np.arange(window - 1, n)
            ok = (((d[ends] - d[ends - (window - 1)]) == window - 1) & 
                  (d[ends] >= lo_i) & (d[ends] <= hi_i) & 
                  (d[ends] + LEAD_DAYS <= d[-1]))
            valid = ends[ok]
            if len(valid) == 0: continue
            
            e = int(rng.choice(valid))
            X.append(v[e - window + 1:e + 1])
            y.append(0)
            end_days.append(d[e]) # Capture exact day for decay calculation

    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])
    return np.array(X), np.array(y), np.array(end_days)

def engineer_tree_features(X_3d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # 1. Absolute states
        last_step = X_3d[:, -1, :]
        mean_val = np.nanmean(X_3d, axis=1)
        max_val = np.nanmax(X_3d, axis=1)
        
        # 2. Volatility (Instability is a massive failure predictor)
        std_val = np.nanstd(X_3d, axis=1)
        
        # 3. Trajectory & Momentum
        delta_val = X_3d[:, -1, :] - X_3d[:, 0, :]
        window = X_3d.shape[1]
        roc_7_days = X_3d[:, -1, :] - X_3d[:, -8, :] if window >= 8 else delta_val
        
        # 4. Moving Average Crossover (Detects sudden recent degradation)
        short_window = max(2, window // 3)
        short_mean = np.nanmean(X_3d[:, -short_window:, :], axis=1)
        long_mean = np.nanmean(X_3d[:, :window - short_window, :], axis=1)
        short_vs_long = short_mean - long_mean

        # 5. Exponential Smoothing (EWMA) matching the paper's methodology
        ewma_features = []
        for alpha in [0.2, 0.5, 0.8]:
            w = (1 - alpha) ** np.arange(window)[::-1]
            w = w / w.sum()
            
            valid_mask = ~np.isnan(X_3d)
            weighted_X = np.where(valid_mask, X_3d * w[np.newaxis, :, np.newaxis], 0.0)
            weight_sum = np.where(valid_mask, w[np.newaxis, :, np.newaxis], 0.0).sum(axis=1)
            ewma = weighted_X.sum(axis=1) / (weight_sum + 1e-9)
            ewma_features.append(ewma)

    # Horizontally stack all arrays
    return np.hstack([last_step, mean_val, max_val, std_val, delta_val, roc_7_days, short_vs_long] + ewma_features)


def evaluate(X_tr, y_tr, X_te, y_te, sample_weight=None):
    m = lgb.LGBMClassifier(
        random_state=SEED, 
        verbose=-1, 
        n_estimators=300,             
        learning_rate=0.05,           
        colsample_bytree=0.8,         
        subsample=0.8,                
        min_child_samples=50          
    )
        
    m.fit(X_tr, y_tr, sample_weight=sample_weight)
    p = m.predict_proba(X_te)[:, 1]
    
    pr_auc = average_precision_score(y_te, p)
    prec, rec, _ = precision_recall_curve(y_te, p)
    
    f1s = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    
    return pr_auc, float(f1s[best_idx]), float(prec[best_idx]), float(rec[best_idx])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    print("Scanning datasets for per-drive stats...")
    stats = get_serial_stats(PARQUET_PATHS)
    data_end = stats["max_date"].max()
    print(f"  {len(stats):,} drives; data ends {data_end}")

    configs = make_configs(data_end)
    fail_days_all = {
        r["serial_number"]: to_int(r["fail_date"])
        for r in stats.filter(pl.col("fail_date").is_not_null())
                      .select(["serial_number", "fail_date"]).iter_rows(named=True)
    }

    summary = []
    for name in CONFIGS_TO_RUN:
        cfg = configs[name]
        print("\n" + "=" * 85)
        print(f"CONFIG: {name}")
        print("=" * 85)
        rng = random.Random(SEED)          
        
        # Load ALL available histories (Natural ratios)
        train_f, train_h_max, test_f, test_h, bounds = select_serials(stats, cfg, rng, train_ratio=None, test_ratio=None)
        print(f"  Train window-end range: {bounds['train'][0]} -> {bounds['train'][1]}")
        print(f"  Test  window-end range: {bounds['test'][0]} -> {bounds['test'][1]}")
        print(f"  Loading histories ONCE for max possible size ({len(train_f)} failed / {len(train_h_max)} healthy)")
        
        train_serials_max = train_f + train_h_max
        test_serials = test_f + test_h
        fail_map = {s: fail_days_all[s] for s in (train_f + test_f)}
        
        print(f"  Pre-computing TRAIN set features in chunks ({len(train_serials_max)} drives)...")
        train_data_max = {w: {"X": [], "y": [], "days": []} for w in WINDOW_SIZES}
        chunk_sz = 20000
        
        for i in range(0, len(train_serials_max), chunk_sz):
            chunk = train_serials_max[i:i + chunk_sz]
            hist_train = load_histories(PARQUET_PATHS, chunk)
            for w in WINDOW_SIZES:
                X_3d, y, days = extract_windows(hist_train, chunk, fail_map, bounds["train"], w)
                if len(y) > 0:
                    train_data_max[w]["X"].append(engineer_tree_features(X_3d))
                    train_data_max[w]["y"].append(y)
                    train_data_max[w]["days"].append(days)
            del hist_train
            gc.collect()
            
        # Combine train chunks into lightweight 2D arrays
        for w in WINDOW_SIZES:
            if train_data_max[w]["y"]:
                train_data_max[w] = (
                    np.vstack(train_data_max[w]["X"]),
                    np.concatenate(train_data_max[w]["y"]),
                    np.concatenate(train_data_max[w]["days"])
                )
            else:
                train_data_max[w] = (np.array([]), np.array([]), np.array([]))
        
        print(f"  Pre-computing TEST set features ({len(test_serials)} drives in chunks)...")
        test_data = {}
        X_te_chunks = {w: [] for w in WINDOW_SIZES}
        y_te_chunks = {w: [] for w in WINDOW_SIZES}
        
        chunk_sz = 20000
        for i in range(0, len(test_serials), chunk_sz):
            chunk = test_serials[i:i + chunk_sz]
            hist_test = load_histories(PARQUET_PATHS, chunk)
            for w in WINDOW_SIZES:
                # Unpack the 3rd variable (days) but ignore it for the test set using _
                X_3d, y, _ = extract_windows(hist_test, chunk, fail_map, bounds["test"], w)
                if len(y) > 0:
                    X_te_chunks[w].append(engineer_tree_features(X_3d))
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

        for ratio in [3, 4, "Natural"]:
            print(f"\n  --- Testing Train Ratio: {ratio} ---")
            
            for w in WINDOW_SIZES:
                X_all, y_all, days_all = train_data_max[w]
                if len(y_all) == 0:
                    continue
                    
                fail_mask = (y_all == 1)
                healthy_mask = (y_all == 0)
                
                if ratio == "Natural":
                    keep_indices = np.arange(len(y_all))
                else:
                    num_fails = fail_mask.sum()
                    num_healthy_needed = num_fails * ratio
                    healthy_indices = np.where(healthy_mask)[0][:num_healthy_needed]
                    fail_indices = np.where(fail_mask)[0]
                    keep_indices = np.concatenate([fail_indices, healthy_indices])
                
                X_tr_2d = X_all[keep_indices]
                y_tr = y_all[keep_indices]
                days_tr = days_all[keep_indices]
                
                if len(y_tr) == 0 or y_tr.sum() == 0:
                    continue
                
                X_te_nat, y_te_nat = test_data[w]
                if len(y_te_nat) == 0 or y_te_nat.sum() == 0:
                    continue
                
                # --- TIME DECAY CALCULATION ---
                cutoff_int = to_int(CUTOFF)
                days_ago = np.maximum(0, cutoff_int - days_tr)
                time_weights = np.power(0.5, days_ago / 180.0)
                
                pos_weight = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
                final_weights = time_weights.copy()
                final_weights[y_tr == 1] *= pos_weight
                
                # --- Evaluate on 50:1 Test Set ---
                # Disabled as per user request to test natural ones only
                # fail_mask_te = (y_te_nat == 1)
                # healthy_mask_te = (y_te_nat == 0)
                # num_fails_te = fail_mask_te.sum()
                # num_healthy_needed_te = num_fails_te * 50
                # 
                # healthy_indices_te = np.where(healthy_mask_te)[0][:num_healthy_needed_te]
                # fail_indices_te = np.where(fail_mask_te)[0]
                # keep_indices_te = np.concatenate([fail_indices_te, healthy_indices_te])
                # 
                # X_te_50 = X_te_nat[keep_indices_te]
                # y_te_50 = y_te_nat[keep_indices_te]
                # 
                # base_50 = y_te_50.mean() if len(y_te_50) > 0 else 0
                # pr_auc_50, f1_50, best_prec_50, best_rec_50 = evaluate(X_tr_2d, y_tr, X_te_50, y_te_50, sample_weight=final_weights)
                # 
                # print(f"  [50:1 Test]    window={w:>2} | train {len(y_tr):>7} | "
                #       f"LGBM (PR-AUC:{pr_auc_50:.4f}, F1:{f1_50:.4f}, Prec:{best_prec_50:.4f}, Rec:{best_rec_50:.4f})")
                # summary.append((name, str(ratio), "50:1", w, int(y_tr.sum()), int(y_te_50.sum()), base_50, pr_auc_50, f1_50, best_prec_50, best_rec_50))
                
                # --- Evaluate on Natural Test Set ---
                base_nat = y_te_nat.mean() if len(y_te_nat) > 0 else 0
                pr_auc_nat, f1_nat, best_prec_nat, best_rec_nat = evaluate(X_tr_2d, y_tr, X_te_nat, y_te_nat, sample_weight=final_weights)
                
                print(f"  [Natural Test] window={w:>2} | train {len(y_tr):>7} | "
                      f"LGBM (PR-AUC:{pr_auc_nat:.4f}, F1:{f1_nat:.4f}, Prec:{best_prec_nat:.4f}, Rec:{best_rec_nat:.4f})")
                summary.append((name, str(ratio), "Natural", w, int(y_tr.sum()), int(y_te_nat.sum()), base_nat, pr_auc_nat, f1_nat, best_prec_nat, best_rec_nat))
                
                del X_tr_2d
                gc.collect()

        del train_data_max, test_data
        gc.collect()

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'config':<22}{'train_R':>8}{'test_R':>8}{'win':>4}{'tr_fail':>9}{'LGB_PR':>9}{'LGB_F1':>8}{'LGB_Prec':>10}{'LGB_Rec':>9}")
    for name, tr_ratio, te_ratio, w, trf, tef, base, pa, f1, best_prec, best_rec in summary:
        print(f"{name:<22}{tr_ratio:>8}{te_ratio:>8}{w:>4}{trf:>9}{pa:>9.4f}{f1:>8.4f}{best_prec:>10.4f}{best_rec:>9.4f}")

if __name__ == "__main__":
    main()