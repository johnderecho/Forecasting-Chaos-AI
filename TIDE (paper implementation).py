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
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import optuna

import dysts
import dysts.metrics

from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE

# ---------------- CONFIG ----------------
input_train_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\train_multivariate__pts_per_period_100__periods_12.json.gz"
input_test_path  = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\test_multivariate__pts_per_period_100__periods_12.json.gz"

output_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\1. Code\results\results_optuna_tide.json"

FORCE_RETRAIN = True
LONG = True                                             # controls split like your prior code
N_TRIALS = 40                                           # recommended: 20-100 depending on time/resources
RANDOM_SEED = 42
PRUNER = optuna.pruners.MedianPruner(n_warmup_steps=5)  # optional pruning
STORAGE = None                                          # e.g., "sqlite:///optuna_study.db" to persist study
# ----------------------------------------

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
def make_tide_objective(attractor_name, train_past, train_future, n_dims, timeout=None):
    """
    Return an objective function for Optuna that uses TiDE.
    The objective returns averaged validation sMAPE across dims (to be minimized).
    """
    def objective(trial):
        # Sample hyperparameters (examples; tune as you prefer)
        # We will sample a set that is broadly safe; unsupported kwargs are filtered later.
        # Numeric ranges chosen conservatively; feel free to change.
        input_size = trial.suggest_int("input_size", 10, max(10, len(train_past)//2))

        # As per gilpin the code is only restricted in varying the "lookback window" per model. Uncomment if relevant
        #hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        #n_layers = trial.suggest_int("n_layers", 1, 4)
        #lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        #dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        #batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        #max_steps = trial.suggest_int("max_steps", 100, 800)  # limit training steps

        # Pack into candidate config
        candidate = {"input_size": int(input_size), "h": len(train_future)}

        # uncomment if other hyperparameters are to be tuned
        # candidate = {
        #    "input_size": int(input_size),
        #    "hidden_size": int(hidden_size),
        #    "num_layers": int(n_layers),
        #    "learning_rate": float(lr),
        #    "dropout": float(dropout),
        #    "batch_size": int(batch_size),
        #    "max_steps": int(max_steps),
        #    # keep horizon consistent for tuning
        #    "h": len(train_future)
        #}

        # Filter candidate to only keys accepted by TiDE.__init__
        sig = inspect.signature(TiDE.__init__)
        candidate_clean = {k: v for k, v in candidate.items() if k in sig.parameters.keys()}

        # Per-dimension training & validation
        per_dim_preds = []
        per_dim_true = []
        for d in range(n_dims):
            # prepare series
            y_train_series = train_past[:, d] if n_dims > 1 else train_past
            y_val_series = train_future[:, d] if n_dims > 1 else train_future

            # build df
            df_train = pd.DataFrame({
                "unique_id": 0,
                "ds": np.arange(len(y_train_series)),
                "y": y_train_series
            })

            # instantiate NeuralForecast with TiDE(**candidate_clean)
            try:
                model = NeuralForecast(models=[TiDE(**candidate_clean)], freq=1)
            except Exception as e:
                # If instantiation fails for this candidate, prune the trial
                trial.set_user_attr("instantiation_error", str(e))
                raise optuna.exceptions.TrialPruned()

            # train
            try:
                model.fit(df_train)
            except Exception as e:
                trial.set_user_attr("fit_error", str(e))
                raise optuna.exceptions.TrialPruned()

            # predict; expected length = len(train_future)
            fcst = model.predict()
            y_pred = extract_forecast_array(fcst, expected_len=len(train_future))
            per_dim_preds.append(y_pred)
            per_dim_true.append(y_val_series)

            # Optional pruning signal: compute partial smape on this dim so far
            # (we only have final predictions here, so use full result)

        # Stack predictions -> shape (N, D) or 1D if univariate
        if n_dims > 1:
            min_len = min(len(p) for p in per_dim_preds)
            per_dim_preds = [p[:min_len] for p in per_dim_preds]
            y_pred_combined = np.column_stack(per_dim_preds)
            y_true_combined = per_dim_true[0][:min_len]
            for i in range(1, len(per_dim_true)):
                y_true_combined = np.column_stack((y_true_combined, per_dim_true[i][:min_len])) if n_dims > 1 else per_dim_true[i][:min_len]
            # careful: per_dim_true constructed as list; rewrite properly:
            y_true_combined = np.column_stack([t[:min_len] for t in per_dim_true])
        else:
            y_pred_combined = per_dim_preds[0]
            y_true_combined = per_dim_true[0]

        # Ensure shapes & compute sMAPE
        y_t, y_p = prepare_for_metrics(y_true_combined, y_pred_combined)
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
    best_params = study.best_trial.user_attrs.get("candidate", None)
    # If candidate not saved, reconstruct from trial.params (best_trial.params) but we filtered keys earlier.
    if best_params is None:
        best_params = dict(study.best_trial.params)
        # ensure horizon replaced:
        best_params["h"] = len(train_future)

    # Save tuning summary
    all_results.setdefault(attractor, {})
    all_results[attractor]["optuna_summary"] = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": len(study.trials)
    }

    # ---------- 2) Retrain the best model on test_past and evaluate ----------
    # Filter params for TiDE signature
    sig = inspect.signature(TiDE.__init__)
    best_clean = {k: v for k, v in best_params.items() if k in sig.parameters.keys()}

    # Per-dim final predictions
    final_preds = []
    final_metrics_per_dim = []

    for d in range(n_dims):
        y_train_series = test_past[:, d] if n_dims > 1 else test_past
        y_true_future = test_future[:, d] if n_dims > 1 else test_future

        df_train = pd.DataFrame({
            "uniGITque_id": 0,
            "ds": np.arange(len(y_train_series)),
            "y": y_train_series
        })

        # Set horizon to length of test_future
        best_clean["h"] = len(test_future)
        try:
            final_model = NeuralForecast(models=[TiDE(**best_clean)], freq=1)
            final_model.fit(df_train)
            fcst_final = final_model.predict()
            y_pred_final = extract_forecast_array(fcst_final, expected_len=len(test_future))
        except Exception as e:
            print("Final training/predict failed for dim", d, ":", e)
            y_pred_final = np.full(len(test_future), np.nan)

        final_preds.append(y_pred_final)
        y_t, y_p = prepare_for_metrics(y_true_future, y_pred_final)
        try:
            metrics = dysts.metrics.compute_metrics(y_t, y_p)
        except Exception as e:
            print("Final metrics failed for dim", d, ":", e)
            metrics = {k: None for k in dysts.metrics.compute_metrics(y_t, y_t).keys()}
        final_metrics_per_dim.append(metrics)

        # Save per-dim
        key_dim = f"TiDE_dim{d}" if n_dims > 1 else "TiDE"
        all_results[attractor].setdefault(key_dim, {})
        all_results[attractor][key_dim]["prediction"] = y_pred_final.tolist()
        all_results[attractor][key_dim]["metrics"] = metrics

    # Combine predictions
    if len(final_preds) > 0:
        min_len = min(len(p) for p in final_preds)
        final_preds = [p[:min_len] for p in final_preds]
        if n_dims > 1:
            y_pred_comb = np.column_stack(final_preds)
        else:
            y_pred_comb = final_preds[0][:min_len]
    else:
        y_pred_comb = np.array([])
        
    # Combined metrics (multivariate)
    try:
        y_true_comb = test_future[:len(y_pred_comb)]
        y_t, y_p = prepare_for_metrics(y_true_comb, y_pred_comb)
        combined_metrics = dysts.metrics.compute_metrics(y_t, y_p)
    except Exception as e:
        print("Combined metrics failed:", e)
        y_dummy = test_future
        if y_dummy.ndim == 1:
            y_dummy = y_dummy.reshape(-1, 1)
        combined_metrics = dysts.metrics.compute_metrics(y_dummy, y_dummy)
        for k in combined_metrics:
            combined_metrics[k] = None

    all_results[attractor]["TiDE_final"] = {
        "prediction": np.array(y_pred_comb).tolist(),
        "combined_metrics": combined_metrics,
        "best_params": best_clean
    }

    # Save after each attractor
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

print("Done. Results saved to", output_path)