#!/usr/bin/python
# optuna_tide_pipeline.py
"""
Optuna-based hyperparameter search and final evaluation for TiDE.

Workflow (per attractor):
1) Load train & test trajectories (from separate gz JSONs).
2) Split train trajectory at t* -> train_past / train_future.
3) Use Optuna to search hyperparameters:
   - For each trial: sample hyperparams, train model(s) on train_past, predict train_future,
     compute validation sMAPE (averaged across dims) -> return to Optuna.
4) After search, get best hyperparameters.
5) Retrain a *single* model (winner) on test_past, forecast test_future, compute final metrics.
6) Save results (tuning summary + final predictions/metrics) to JSON.

Notes:
- This script only implements TiDE in the search, but is structured so you can add other models later.
- It uses a safe parameter filtering step before instantiating the model,
  so Optuna can propose many candidates but only supported args are passed.
"""

import os
import gzip
import json
import inspect
import time
from typing import Dict, Any
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.preprocessing import StandardScaler

import dysts
import dysts.metrics

from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE

# ---------------- CONFIG ----------------
input_train_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\train_multivariate__pts_per_period_100__periods_12.json.gz"
input_test_path  = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\test_multivariate__pts_per_period_100__periods_12.json.gz"

output_path = r"C:\Users\Windows\Desktop\Thesis - GPU\results\results_optuna_tide.json"

FORCE_RETRAIN = True
LONG = True                                             # controls split like your prior code
N_TRIALS = 1                                           # recommended: 20-100 depending on time/resources
RANDOM_SEED = 42
PRUNER = optuna.pruners.MedianPruner(n_warmup_steps=5)  # optional pruning
STORAGE = None                                          # e.g., "sqlite:///optuna_study.db" to persist study
# ----------------------------------------

start_time = time.time()


torch.set_float32_matmul_precision('medium')
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")





# ---------- Helpers ----------
def load_gz_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)

def extract_forecast_array(fcst_df, expected_len):
    # Try common names
    for col in ("y_hat", "y", "prediction"):
        if col in fcst_df.columns:
            arr = fcst_df[col].values
            return np.asarray(arr).reshape(-1)
    for c in fcst_df.columns:
        if c not in ("unique_id", "ds"):
            arr = fcst_df[c].values
            if len(arr) == expected_len:
                return np.asarray(arr).reshape(-1)
    return np.full(expected_len, np.nan)

def prepare_for_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    min_rows = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:min_rows, :]
    y_pred = y_pred[:min_rows, :]
    min_cols = min(y_true.shape[1], y_pred.shape[1])
    y_true = y_true[:, :min_cols]
    y_pred = y_pred[:, :min_cols]
    return y_true, y_pred

def smape_per_dim(y_true, y_pred):
    """
    Compute sMAPE per-dimension and return mean across dims.
    y_true, y_pred are numpy arrays (N,D) or (N,) accepted.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1,1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1,1)
    # align
    min_rows = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:min_rows]
    y_pred = y_pred[:min_rows]
    # Now compute dim-wise smape
    eps = 1e-8
    dim_smapes = []
    for d in range(y_true.shape[1]):
        a = y_true[:, d]
        b = y_pred[:, d]
        denom = (np.abs(a) + np.abs(b) + eps)
        s = 2.0 * np.abs(a - b) / denom
        dim_smapes.append(np.mean(s) * 100.0)  # percent
    return float(np.mean(dim_smapes))


def create_multivariate_df(data_array, start_idx=0):
    """
    Creates a NeuralForecast compliant DataFrame where every unique_id
    has access to all other dimensions via exogenous columns.
    
    data_array: shape (Time, Dims)
    """
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)
        
    n_time, n_dims = data_array.shape
    df_list = []
    
    # Create the exogenous columns names
    exog_cols = [f"var_{d}" for d in range(n_dims)]
    
    for d in range(n_dims):
        # Base dict for this dimension
        tmp_dict = {
            "unique_id": d,
            "ds": np.arange(start_idx, start_idx + n_time),
            "y": data_array[:, d]
        }
        # Add all dimensions as exogenous columns
        for dim_idx, col_name in enumerate(exog_cols):
            tmp_dict[col_name] = data_array[:, dim_idx]
            
        df_list.append(pd.DataFrame(tmp_dict))
        
    return pd.concat(df_list).reset_index(drop=True), exog_cols

def recursive_rollout(nf_model, initial_history, n_steps_to_predict, n_dims, model_h, scaler):
    """
    Performs closed-loop rollout. 
    1. Predicts model_h steps.
    2. Appends predictions to history.
    3. Repeats until n_steps_to_predict is covered.
    
    Returns: Unscaled predictions of shape (n_steps_to_predict, n_dims)
    """
    current_history = initial_history.copy() # (T, D)
    preds_accumulated = []
    
    steps_generated = 0
    
    while steps_generated < n_steps_to_predict:
        # Create input DF from recent history (enough to cover input_size)
        # We pass the WHOLE history; TiDE will slice the end automatically based on input_size
        df_in, _ = create_multivariate_df(current_history, start_idx=0)
        
        # Predict next window
        # NeuralForecast predict uses the end of df_in as the forecast point
        fcst = nf_model.predict(df_in)
        
        # Extract predictions for this window
        pred_col = [c for c in fcst.columns if c not in ["unique_id", "ds", "y"]][0]
        preds_df = fcst.pivot(index="ds", columns="unique_id", values=pred_col).sort_index(axis=1)
        
        # Shape: (model_h, n_dims)
        new_preds = preds_df.values 
        
        # Append to history for next iteration (Closed-Loop)
        current_history = np.vstack([current_history, new_preds])
        preds_accumulated.append(new_preds)
        
        steps_generated += len(new_preds)

    # Concatenate and trim to exact required length
    full_pred_scaled = np.vstack(preds_accumulated)
    if len(full_pred_scaled) > n_steps_to_predict:
        full_pred_scaled = full_pred_scaled[:n_steps_to_predict]
        
    # Inverse transform
    return scaler.inverse_transform(full_pred_scaled)


# ------------ Load data ------------
train_eq = load_gz_json(input_train_path)
test_eq  = load_gz_json(input_test_path)

try:
    with open(output_path, "r") as f:
        all_results = json.load(f)
except FileNotFoundError:
    all_results = {}

# Basic model registry (only TiDE for now)
NF_MODELS = {"TiDE": TiDE}

# Attractor selection (adapt to your preference)
TRAIN_ALL = False
TARGET_ATTRACTOR = "Aizawa"
if TRAIN_ALL:
    attractor_list = list(train_eq.keys())
else:
    if TARGET_ATTRACTOR not in train_eq:
        raise ValueError(f"{TARGET_ATTRACTOR} not found in training data")
    attractor_list = [TARGET_ATTRACTOR]


# --------------- Optuna objective for TiDE ----------------
def make_tide_objective(attractor_name, train_past_raw, train_future_raw, n_dims, timeout=None):
    # 1. Standardization (fit on train_past only)
    scaler = StandardScaler()                              # <── keep a local scaler
    train_past_scaled = scaler.fit_transform(
        train_past_raw if n_dims > 1 else train_past_raw.reshape(-1, 1)
    )

    def objective(trial):
        print(f"Running Trial {trial.number}")
        
        # Sample hyperparameters
        # We introduce 'model_h' (prediction horizon for one forward pass)
        # Closed-loop rollout means we train on short h, predict long sequence recursively.
        model_h = trial.suggest_int("model_h", 20, 100) 
        input_size = trial.suggest_int("input_size", model_h, max(model_h, 200))

        # Pack into candidate config
        # NOTE: to match original one-shot methodology we set h = full horizon
        candidate = {
            "input_size": int(input_size),
            "h": len(train_future_raw),          # one-shot prediction
            "hist_exog_list": [f"var_{d}" for d in range(n_dims)]
        }

        # Filter candidate to only keys accepted by TiDE.__init__
        sig = inspect.signature(TiDE.__init__)
        candidate_clean = {k: v for k, v in candidate.items() if k in sig.parameters.keys()}

        # Prepare Training Data (Scaled)
        # We need to construct the DF with exogenous columns
        df_train, exog_cols = create_multivariate_df(train_past_scaled)

        try:
            # Instantiate and fit
            model = NeuralForecast(models=[TiDE(**candidate_clean)], freq=1)
            model.fit(df_train)

            # One-shot prediction
            fcst_val = model.predict()
            pred_col = [c for c in fcst_val.columns if c not in ["unique_id", "ds", "y"]][0]
            preds_val = (
                fcst_val.pivot(index="ds", columns="unique_id", values=pred_col)
                        .reindex(columns=range(n_dims))
                        .sort_index()
            ).values
            y_pred_original_scale = scaler.inverse_transform(preds_val)

            # ground truth for validation
            y_true_combined = train_future_raw               # <── add this line

        except Exception as e:
            # If instantiation/fit/predict fails for this candidate, prune the trial
            trial.set_user_attr("error", str(e))
            raise optuna.exceptions.TrialPruned()

        # Ensure shapes & compute sMAPE
        y_t, y_p = prepare_for_metrics(y_true_combined, y_pred_original_scale)
        val_smape = smape_per_dim(y_t, y_p)

        # Report intermediate value to Optuna (for pruning)
        trial.report(val_smape, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Save trial attributes for debugging
        trial.set_user_attr("candidate", candidate_clean)
        trial.set_user_attr("val_smape", float(val_smape))

        return float(val_smape)

    return objective


# ----------------- Main loop -----------------
for attractor in attractor_list:
    print(f"\n=== Attractor: {attractor} ===")

    if attractor not in train_eq:
        print("Missing attractor in train file, skipping")
        continue
    if attractor not in test_eq:
        print("Missing attractor in test file, skipping")
        continue

    train_traj = np.array(train_eq[attractor]["values"])
    test_traj  = np.array(test_eq[attractor]["values"])

    # t* split
    split_point = int(5 / 6 * len(train_traj))
    if LONG:
        split_point = int(1 / 6 * len(train_traj))

    train_past = train_traj[:split_point]
    train_future = train_traj[split_point:]

    test_past = test_traj[:split_point]
    test_future = test_traj[split_point:]

    n_dims = 1 if train_past.ndim == 1 else train_past.shape[1]

    # ---------- 1) Optuna search ----------
    study_name = f"optuna_tide_{attractor}"
    if STORAGE is None:
        study = optuna.create_study(direction="minimize", pruner=PRUNER, study_name=study_name)
    else:
        study = optuna.create_study(direction="minimize", pruner=PRUNER, study_name=study_name, storage=STORAGE, load_if_exists=True)

    objective = make_tide_objective(attractor, train_past, train_future, n_dims)
    print(f"Starting Optuna search for {attractor} (trials={N_TRIALS}) ...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("Best trial:", study.best_trial.number, "value (val smape):", study.best_trial.value)
    
    # Retrieve best params and ensure exogenous list is present
    best_params = study.best_trial.user_attrs.get("candidate", None)
    if best_params is None:
        best_params = dict(study.best_trial.params)
        # Note: In fallback, we might miss hist_exog_list if it wasn't in params. Re-add it:
        best_params["hist_exog_list"] = [f"var_{d}" for d in range(n_dims)]
        # Also ensure 'h' is set if we fell back to raw params which used 'model_h' key
        if "h" not in best_params and "model_h" in best_params:
            best_params["h"] = best_params["model_h"]

    # Save tuning summary
    all_results.setdefault(attractor, {})
    all_results[attractor]["optuna_summary"] = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": len(study.trials)
    }

    # ---------- 2) Retrain the best model on test_past and evaluate ----------
    
    # Re-Fit Scaler on Test Past (Standardization per variable for final run)
    # Note: Standard practice in time series can vary, but typically we fit scaler on available history.
    scaler = StandardScaler()
    test_past_scaled = scaler.fit_transform(test_past if n_dims > 1 else test_past.reshape(-1, 1))

    # Filter params for TiDE signature
    sig = inspect.signature(TiDE.__init__)
    best_clean = {k: v for k, v in best_params.items() if k in sig.parameters.keys()}
    
    # CRITICAL: Do NOT set h = len(test_future) here. 
    # We use the optimized 'model_h' (short horizon) and do closed-loop rollout.
    model_h = best_clean["h"]

    # Prepare data (test_past) for all dims with exogenous support
    df_train, _ = create_multivariate_df(test_past_scaled)

    y_pred_comb = np.array([])
    final_metrics_per_dim = []

    try:
        final_model = NeuralForecast(models=[TiDE(**best_clean)], freq=1)
        final_model.fit(df_train)
        
        # Predict next window
        fcst = final_model.predict()

        pred_col = [c for c in fcst.columns if c not in ["unique_id", "ds", "y"]][0]
        preds_df = (
            fcst.pivot(index="ds", columns="unique_id", values=pred_col)
                .reindex(columns=range(n_dims))    # enforce X-Y-Z order
                .sort_index()
        )

        y_pred_comb = scaler.inverse_transform(preds_df.values)   # back to physical units

    except Exception as e:
        print("Final training/predict failed:", e)
        # Fallback for metrics calculation to prevent crash
        import traceback
        traceback.print_exc()

# ---------- final retrain section ------------------------------------
# we already fitted a 'scaler' on test_past_scaled just above
try:
    final_model = NeuralForecast(models=[TiDE(**best_clean)], freq=1)
    final_model.fit(df_train)

    fcst_final = final_model.predict()
    pred_col = [c for c in fcst_final.columns if c not in ["unique_id", "ds", "y"]][0]
    preds_df = (
        fcst_final.pivot(index="ds", columns="unique_id", values=pred_col)
                 .reindex(columns=range(n_dims))   # enforce axis order
                 .sort_index()
    )
    y_pred_comb = scaler.inverse_transform(preds_df.values)  # use test-set scaler
    if len(y_pred_comb) > len(test_future):
        y_pred_comb = y_pred_comb[:len(test_future)]
except Exception as e:
    print("Final training/predict failed:", e)
# ----------------------------------------------------------------------