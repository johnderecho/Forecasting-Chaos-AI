#!/usr/bin/python
# optuna_TSMixer_pipeline_clean.py

import os
import gzip
import json
import inspect
import time
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import optuna

import dysts
import dysts.metrics

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

# ================= CONFIG =================
input_train_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\train_multivariate__pts_per_period_100__periods_12.json.gz"   # use dynamic finder
input_test_path  = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\test_multivariate__pts_per_period_100__periods_12.json.gz"    # use dynamic finder
output_path = r"C:\Users\Windows\Desktop\Thesis - GPU\results\Transformer PatchTST Base - 10.json"                                                             # use dynamic finder

TARGET_ATTRACTOR = "Lorenz"
TRAIN_ALL = False

LONG = True
N_TRIALS = 1                                                                                                           # Number of trials for hyperparameter optimization
RANDOM_SEED = 42
n_series = 3                                                                                                               # Number of series to train on TSMixer_final

PRUNER = optuna.pruners.NopPruner()   # pruning disabled (safe)
STORAGE = None

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))



# ================= HELPERS =================

def load_gz_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)

def prepare_for_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n = min(len(y_true), len(y_pred))
    d = min(y_true.shape[1], y_pred.shape[1])

    return y_true[:n, :d], y_pred[:n, :d]

def smape_per_dim(y_true, y_pred):
    eps = 1e-8
    smapes = []
    for d in range(y_true.shape[1]):
        a = y_true[:, d]
        b = y_pred[:, d]
        s = 2 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)
        smapes.append(np.mean(s) * 100)
    return float(np.mean(smapes))



# ================= LOAD DATA =================

train_eq = load_gz_json(input_train_path)
test_eq  = load_gz_json(input_test_path)

if TRAIN_ALL:
    attractors = list(train_eq.keys())
else:
    attractors = [TARGET_ATTRACTOR]

try:
    with open(output_path, "r") as f:
        all_results = json.load(f)
except FileNotFoundError:
    all_results = {}



# ================= OPTUNA OBJECTIVE =================

def make_objective(train_past, train_future, n_dims):

    def objective(trial):

        input_size = trial.suggest_int(
            "input_size",
            10,
            max(10, len(train_past) // 2)
        )

        params = {
            "input_size": input_size,
            "h": len(train_future),
            "n_series": n_series
        }

        # Filter only valid PatchTST args
        sig = inspect.signature(PatchTST.__init__)
        params = {k: v for k, v in params.items() if k in sig.parameters}

        # Build global dataframe
        dfs = []
        for d in range(n_dims):
            series = train_past[:, d] if n_dims > 1 else train_past
            dfs.append(pd.DataFrame({
                "unique_id": d,
                "ds": np.arange(len(series)),
                "y": series
            }))
        df_train = pd.concat(dfs).reset_index(drop=True)

        try:
            model = NeuralForecast(models=[PatchTST(**params)], freq=1)
            model.fit(df_train)
            fcst = model.predict()

            pred_col = [c for c in fcst.columns if c not in ("unique_id", "ds", "y")][0]
            preds = (
                fcst.pivot(index="ds", columns="unique_id", values=pred_col)
                    .reindex(columns=range(n_dims))
                    .sort_index()
                    .values
            )

            y_true = train_future

        except Exception as e:
            trial.set_user_attr("error", str(e))
            raise  # FAIL the trial, do not prune

        y_t, y_p = prepare_for_metrics(y_true, preds)
        smape = smape_per_dim(y_t, y_p)

        trial.set_user_attr("params", params)
        trial.set_user_attr("val_smape", smape)

        return smape

    return objective



# ==================================
# MAIN LOOP
# ==================================

start = time.time()

for attractor in attractors:

    print(f"\n=== {attractor} ===")

    train_traj = np.asarray(train_eq[attractor]["values"])
    test_traj  = np.asarray(test_eq[attractor]["values"])

    split = int(len(train_traj) / 6) if LONG else int(5 * len(train_traj) / 6)

    train_past   = train_traj[:split]
    train_future = train_traj[split:]

    test_past    = test_traj[:split]
    test_future  = test_traj[split:]

    n_dims = 1 if train_past.ndim == 1 else train_past.shape[1]

    study = optuna.create_study(
        direction="minimize",
        pruner=PRUNER,
        study_name=f"PatchTST_{attractor}",
        storage=STORAGE,
        load_if_exists=True
    )

    study.optimize(
        make_objective(train_past, train_future, n_dims),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No successful trials.")
        continue

    best = study.best_trial
    best_params = best.user_attrs["params"]

    all_results.setdefault(attractor, {})
    all_results[attractor]["optuna"] = {
        "best_value": best.value,
        "best_params": best_params,
        "n_trials": len(study.trials)
    }

    # ===== FINAL TRAINING =====

    best_params["h"] = len(test_future)

    dfs = []
    for d in range(n_dims):
        series = test_past[:, d] if n_dims > 1 else test_past
        dfs.append(pd.DataFrame({
            "unique_id": d,
            "ds": np.arange(len(series)),
            "y": series
        }))
    df_train = pd.concat(dfs).reset_index(drop=True)

    model = NeuralForecast(models=[PatchTST(**best_params)], freq=1)
    model.fit(df_train)

    fcst = model.predict()
    pred_col = [c for c in fcst.columns if c not in ("unique_id", "ds", "y")][0]

    preds = (
        fcst.pivot(index="ds", columns="unique_id", values=pred_col)
            .reindex(columns=range(n_dims))
            .sort_index()
            .values
    )

    preds = preds[:len(test_future)]

    y_t, y_p = prepare_for_metrics(test_future, preds)

    all_results[attractor]["PatchTST_final"] = {
        "prediction": preds.tolist(),
        "metrics": dysts.metrics.compute_metrics(y_t, y_p),
        "best_params": best_params
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

print("\n✅ Done")
print(f"Total time: {time.time() - start:.2f}s")



# ==================================
# Visualization
# ==================================

import os
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- CONFIG ----------------
RESULTS_PATH = r"C:\Users\Windows\Desktop\Thesis - GPU\results\30.json"
DATA_PATH = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\test_multivariate__pts_per_period_100__periods_12.json.gz"
LONG = True

def ensure_3cols(array, name):
    """
    Ensures the array has at least 3 columns for 3D plotting.
    """
    array = np.array(array)
    if array.ndim == 1:
        if array.size < 3:
            raise ValueError(f"{name} too short for 3D plotting (needs >=3 elements)")
        array = array.reshape(-1, 3)
    elif array.ndim == 2 and array.shape[1] < 3:
        last_col = array[:, -1:]
        while array.shape[1] < 3:
            array = np.hstack([array, last_col])
    return array

def plot_phase_space_3d(y_true, y_pred=None, attractor_name=""):
    """
    Plots 3D phase space.
    If y_pred is None, plots only ground truth.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ground truth
    ax.plot(y_true[:, 0], y_true[:, 1], y_true[:, 2],
            label='Ground Truth', color='black', alpha=0.35, linewidth=1.0)

    # Optional prediction
    if y_pred is not None:
        ax.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2],
                label='PatchTST Prediction', color='crimson', alpha=0.9,
                linewidth=1.5, linestyle='--')

    ax.set_title(f"3D Phase Space: {attractor_name}", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show(block=True)

def main():
    # 1) Load Results
    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found at: {RESULTS_PATH}")
        return
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    if not results:
        print("Results JSON is empty or has no attractors.")
        return

    # 2) Load Ground Truth
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at: {DATA_PATH}")
        return
    with gzip.open(DATA_PATH, "rt") as f:
        data_json = json.load(f)

    any_plotted = False
    for attractor_name, attractor_res in results.items():
        print(f"\nProcessing {attractor_name}...")

        if "PatchTST_final" not in attractor_res:
            print(f"  No 'PatchTST_final' for {attractor_name}, skipping.")
            continue
        if attractor_name not in data_json:
            print(f"  No ground truth for {attractor_name}, skipping.")
            continue

        # Ground truth
        full_series = np.array(data_json[attractor_name]["values"])
        n_total = len(full_series)

        split_point = int(5 / 6 * n_total)
        if LONG:
            split_point = int(1 / 6 * n_total)
        y_true = full_series[split_point:]

        # Predictions
        y_pred = np.array(attractor_res["PatchTST_final"]["prediction"])

        if y_pred.size == 0:
            print(f"  Empty prediction for {attractor_name}, skipping.")
            continue

        # Ensure equal lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # Ensure arrays have 3 columns
        try:
            y_true = ensure_3cols(y_true, f"{attractor_name} y_true")
            y_pred = ensure_3cols(y_pred, f"{attractor_name} y_pred")
        except ValueError as e:
            print(f"  Skipping {attractor_name}: {e}")
            continue

        print(f"  y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

        # --- Plot 1: Ground Truth only ---
        plot_phase_space_3d(y_true, y_pred=None, attractor_name=f"{attractor_name} (Ground Truth)")

        # --- Plot 2: Ground Truth + Prediction ---
        plot_phase_space_3d(y_true, y_pred=y_pred, attractor_name=f"{attractor_name} (GT + Prediction)")

        any_plotted = True

    if not any_plotted:
        print("No plots were produced. Check your data and predictions.")

if __name__ == "__main__":
    main()