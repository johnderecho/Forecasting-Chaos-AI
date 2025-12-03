#!/usr/bin/python
# corrected_training_pipeline.py
"""
Workflow:
1) Load TRAIN and TEST trajectories for each attractor.
2) For each attractor:
   a) Split train trajectory at t*: train_past / train_future
   b) Use train_past to fit models and evaluate on train_future (for hyperparam selection).
   c) Choose best hyperparameters (here we load them from JSON already; script trains once per hyperparam entry).
   d) Retrain chosen model on test_past (the past portion of the test trajectory).
   e) Forecast test_future and compute metrics (multivariate-safe).
3) Save results to output JSON.
"""

import os
import json
import gzip
import warnings
import inspect

import numpy as np
import pandas as pd
import torch

import dysts
import dysts.metrics

from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE

# --------- Configuration (EDIT PATHS) ----------
# Provide BOTH train and test dataset files (author supplied two files).
input_train_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\train_multivariate__pts_per_period_100__periods_12.json.gz"
input_test_path  = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\test_multivariate__pts_per_period_100__periods_12.json.gz"

hyperparameter_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\1. Code\hyperparameters\hyperparameters_multivariate_train_multivariate__pts_per_period_100__periods_12.json"
output_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\1. Code\results\results_train_test_multivariate.json"

FORCE_RETRAIN = True
LONG = True  # set forecasting horizon mode; kept from your code
# ------------------------------------------------

torch.set_float32_matmul_precision('medium')         # high is higher performance but affects accuracy
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load data
def load_gz_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)

train_equation_data = load_gz_json(input_train_path)
test_equation_data = load_gz_json(input_test_path)

with open(hyperparameter_path, "r") as f:
    all_hyperparameters = json.load(f)

try:
    with open(output_path, "r") as f:
        all_results = json.load(f)
except FileNotFoundError:
    all_results = dict()

print("Train keys:", list(train_equation_data.keys()))
print("Test keys:", list(test_equation_data.keys()))

# Model registry
NF_MODELS = {"TiDE": TiDE}

# Which attractor(s) to run
TRAIN_ALL = False
TARGET_ATTRACTOR = "Aizawa"
if TRAIN_ALL:
    attractor_list = list(train_equation_data.keys())
else:
    if TARGET_ATTRACTOR not in train_equation_data:
        raise ValueError(f"{TARGET_ATTRACTOR} not found in training data")
    attractor_list = [TARGET_ATTRACTOR]

# Utility: safe extraction of forecast column from NeuralForecast output
def extract_forecast_array(fcst_df, expected_len):
    # NeuralForecast predict() returns a DataFrame with 'unique_id','ds','y', and usually 'y_hat' or model-name column
    # Try common names in priority
    for col in ("y_hat", "y", "prediction",):
        if col in fcst_df.columns:
            arr = fcst_df[col].values
            return np.asarray(arr).reshape(-1)
    # If model name present as a column, use that
    for c in fcst_df.columns:
        if c not in ("unique_id", "ds"):
            arr = fcst_df[c].values
            if len(arr) == expected_len:
                return np.asarray(arr).reshape(-1)
    # fallback: return NaNs of expected length
    return np.full(expected_len, np.nan)

# Ensure numeric arrays and consistent shapes for metrics
def prepare_for_metrics(y_true, y_pred):
    """
    Convert inputs to 2D arrays with shape (N, D).
    Trim to matching min length and min dims.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Ensure 2D: shape (N, D)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Trim rows to min length
    min_rows = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:min_rows, :]
    y_pred = y_pred[:min_rows, :]

    # Trim columns to min dims
    min_cols = min(y_true.shape[1], y_pred.shape[1])
    y_true = y_true[:, :min_cols]
    y_pred = y_pred[:, :min_cols]

    return y_true, y_pred

# Main loop
for attractor in attractor_list:
    print(f"\n=== Attractor: {attractor} ===")
    # load trajectories (author provided separate train/test sets)
    if attractor not in train_equation_data:
        print(f"{attractor} missing in train file, skipping.")
        continue
    if attractor not in test_equation_data:
        print(f"{attractor} missing in test file, skipping.")
        continue

    train_traj = np.array(train_equation_data[attractor]["values"])  # shape (T, ) or (T, D)
    test_traj  = np.array(test_equation_data[attractor]["values"])

    if attractor not in all_results:
        all_results[attractor] = {}

    # Choose split point t* (index). Replicates your LONG handling.
    split_point = int(5 / 6 * len(train_traj))
    if LONG:
        split_point = int(1 / 6 * len(train_traj))

    # Split train trajectory: used for hyperparam selection
    train_past = train_traj[:split_point]
    train_future = train_traj[split_point:]  # used for model selection

    # Split test trajectory: used for final evaluation
    # Use same split ratio (t*) on the test trajectory
    test_split_point = split_point  # same t* index
    test_past = test_traj[:test_split_point]
    test_future = test_traj[test_split_point:]

    # Save the true test_future values (for record)
    all_results[attractor]["values_test_future"] = np.squeeze(test_future).tolist()

    # Loop hyperparameter entries for this attractor
    if attractor not in all_hyperparameters:
        print(f"No hyperparameters for {attractor}, skipping.")
        continue

    for model_name in all_hyperparameters[attractor].keys():
        print(f"Model: {model_name}")
        if (model_name in all_results[attractor].keys()) and not FORCE_RETRAIN:
            print("Skipping (already present)")
            continue

        # Prepare hyperparameters
        pc = dict(all_hyperparameters[attractor][model_name])
        # forecast horizon = length of train_future when tuning
        h_tune = len(train_future)
        pc["h"] = h_tune

        # Special TiDE handling
        if model_name == "TiDE":
            if "input_size" not in pc:
                pc["input_size"] = min(max(h_tune * 2, 10), max(1, len(train_past) // 3))
            else:
                pc["input_size"] = min(pc["input_size"], max(1, len(train_past) - 1))
            pc["max_steps"] = 500

        # filter valid kwargs for model class
        if model_name not in NF_MODELS:
            print(f"Unknown model {model_name}, skipping.")
            continue
        ModelClass = NF_MODELS[model_name]
        sig = inspect.signature(ModelClass.__init__)
        pc_clean = {k: v for k, v in pc.items() if k in sig.parameters.keys()}

        # ---------- 1) Model selection stage (train on train_past, evaluate on train_future) ----------
        # We'll train one instance per hyperparam entry (you could add CV or multiple seeds if desired)

        # Handle multivariate vs univariate
        if train_past.ndim == 1:
            n_steps, n_dims = len(train_past), 1
        else:
            n_steps, n_dims = train_past.shape

        # Collect per-dimension predictions for tuning metrics
        tune_preds_per_dim = []
        tune_metrics_per_dim = []

        for d in range(n_dims):
            y_train_series = train_past[:, d] if n_dims > 1 else train_past
            # DataFrame for training the univariate model
            df_train = pd.DataFrame({
                "unique_id": 0,
                "ds": np.arange(len(y_train_series)),
                "y": y_train_series
            })
            # Ensure horizon correct for tuning
            pc_clean["h"] = len(train_future)
            # instantiate model
            model = NeuralForecast(models=[ModelClass(**pc_clean)], freq=1)
            # train
            model.fit(df_train)
            # predict: NeuralForecast predict on the next h points relative to training df
            fcst = model.predict()
            # extract prediction array
            y_pred_single = extract_forecast_array(fcst, expected_len=len(train_future))
            tune_preds_per_dim.append(y_pred_single)

            # compute metrics vs train_future for this dim
            y_true_dim = train_future[:, d] if n_dims > 1 else train_future
            # align & shape
            y_t, y_p = prepare_for_metrics(y_true_dim, y_pred_single)
            try:
                metrics = dysts.metrics.compute_metrics(y_t, y_p)
            except Exception as e:
                print("Tuning metrics failed for dim", d, ":", e)
                metrics = {k: None for k in dysts.metrics.compute_metrics(y_t, y_t).keys()}
            tune_metrics_per_dim.append(metrics)

        # collapse tune metrics across dims (here we just keep them per-dim and compute a simple average per metric)
        # convert list of dicts to dict of averaged values (ignoring None)
        avg_tune_metrics = {}
        metric_keys = tune_metrics_per_dim[0].keys() if tune_metrics_per_dim else []
        for k in metric_keys:
            vals = [m[k] for m in tune_metrics_per_dim if m.get(k) is not None]
            avg_tune_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else None

        # store tuning info
        all_results[attractor][model_name] = {
            "tuning_metrics": avg_tune_metrics,
            "tune_predictions_per_dim": [p.tolist() for p in tune_preds_per_dim]
        }

        # ---------- 2) Final training & forecast stage (retrain on test_past, predict test_future) ----------
        # Now retrain the model on test_past and predict the test_future
        final_preds_per_dim = []
        final_metrics_per_dim = []

        for d in range(n_dims):
            y_train_series = test_past[:, d] if n_dims > 1 else test_past
            df_train = pd.DataFrame({
                "unique_id": 0,
                "ds": np.arange(len(y_train_series)),
                "y": y_train_series
            })
            # Set horizon to length of test_future
            pc_clean["h"] = len(test_future)
            final_model = NeuralForecast(models=[ModelClass(**pc_clean)], freq=1)
            final_model.fit(df_train)
            fcst_final = final_model.predict()
            y_pred_final = extract_forecast_array(fcst_final, expected_len=len(test_future))
            final_preds_per_dim.append(y_pred_final)

            # Metrics for this dimension
            y_true_dim = test_future[:, d] if n_dims > 1 else test_future
            y_t, y_p = prepare_for_metrics(y_true_dim, y_pred_final)
            try:
                metrics = dysts.metrics.compute_metrics(y_t, y_p)
            except Exception as e:
                print("Final metrics failed for dim", d, ":", e)
                metrics = {k: None for k in dysts.metrics.compute_metrics(y_t, y_t).keys()}
            final_metrics_per_dim.append(metrics)

            # Save per-dim result under key
            key_dim = f"{model_name}_dim{d}" if n_dims > 1 else model_name
            all_results[attractor].setdefault(key_dim, {})
            all_results[attractor][key_dim]["prediction"] = y_pred_final.tolist()
            all_results[attractor][key_dim]["metrics"] = metrics

        # Combine predictions into (N, D) array for multivariate metrics
        if len(final_preds_per_dim) > 0:
            min_len = min(len(p) for p in final_preds_per_dim)
            final_preds_per_dim = [p[:min_len] for p in final_preds_per_dim]
            if n_dims > 1:
                y_val_pred_combined = np.column_stack(final_preds_per_dim)
            else:
                y_val_pred_combined = final_preds_per_dim[0][:min_len]
        else:
            y_val_pred_combined = np.array([])

        # Compute combined multivariate metrics
        try:
            y_true_final = test_future[:len(y_val_pred_combined)]
            # ensure consistent 2D shapes
            y_t, y_p = prepare_for_metrics(y_true_final, y_val_pred_combined)
            combined_metrics = dysts.metrics.compute_metrics(y_t, y_p)
        except Exception as e:
            print("Combined metrics failed:", e)
            y_dummy = test_future
            if y_dummy.ndim == 1:
                y_dummy = y_dummy.reshape(-1, 1)
            combined_metrics = dysts.metrics.compute_metrics(y_dummy, y_dummy)
            for k in combined_metrics:
                combined_metrics[k] = None

        all_results[attractor][model_name]["prediction"] = np.array(y_val_pred_combined).tolist()
        all_results[attractor][model_name]["combined_metrics"] = combined_metrics

        # Save after each model
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=4)

print("Done. Results saved to", output_path)