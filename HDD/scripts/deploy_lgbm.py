import joblib
import numpy as np
import lightgbm as lgb
import json

# The 22 features we used for training
FEATURES = [
    'smart_200_raw', 'smart_197_raw', 'smart_187_raw', 'smart_5_raw', 
    'smart_196_raw', 'smart_2_raw', 'smart_199_raw', 'smart_12_raw', 
    'smart_9_raw', 'smart_188_raw', 'smart_195_raw', 'smart_190_raw', 
    'smart_241_raw', 'smart_198_raw', 'smart_8_raw',
    'smart_192_raw', 'smart_193_raw', 'smart_3_raw', 'smart_7_raw', 
    'smart_194_raw', 'smart_4_raw', 'smart_222_raw'
]

# We save the strict threshold into a config file so your production system knows what to use
config = {
    "model_file": "lgbm_hdd_lead7_window28.pkl",
    "threshold": 0.95,
    "lead_days": 7,
    "window_size": 28,
    "features": FEATURES
}

with open("lgbm_production_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Saved production configuration to 'lgbm_production_config.json' with Threshold set to 0.95")

def simulate_production_inference(model_path, threshold):
    # 1. Load the trained model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # 2. Simulate getting a 28-day window of SMART data for a single hard drive
    # (Shape: 1 drive, 28 days, 22 features)
    dummy_data_3d = np.zeros((1, 28, 22)) 
    
    # 3. Engineer the tree features (max, mean, delta, last_step) exactly like training
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        last_step = dummy_data_3d[:, -1, :]
        mean_val = np.nanmean(dummy_data_3d, axis=1)
        delta_val = dummy_data_3d[:, -1, :] - dummy_data_3d[:, 0, :]
        max_val = np.nanmax(dummy_data_3d, axis=1)
    
    X_inference = np.hstack([last_step, mean_val, delta_val, max_val])
    
    # 4. Predict probability
    probability_of_failure = model.predict_proba(X_inference)[0][1]
    
    # 5. Apply our strict 0.95 threshold
    if probability_of_failure >= threshold:
        print(f"ALERT! Drive failure predicted. (Confidence: {probability_of_failure:.2f} >= {threshold}) -> Triggering physical replacement.")
    else:
        print(f"Drive is healthy. (Confidence: {probability_of_failure:.2f} < {threshold}) -> Doing nothing.")

if __name__ == "__main__":
    simulate_production_inference(config["model_file"], config["threshold"])
