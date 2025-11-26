#!/usr/bin/python

import sys
import os
import json
import warnings

import dysts
import dysts_data
from dysts_data import *
import dysts.metrics

import numpy as np
import torch
import pandas as pd
import inspect

from dysts_data import dataloader
from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE
from neuralforecast.losses.pytorch import SMAPE

torch.set_float32_matmul_precision('medium')

FORCE_RETRAIN = True

# GPU Sanity Check
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# =======================================================
# 2. Paths and Configs
# =======================================================
LONG = True

# Replace "path/to/your/directory" with the actual path to your working directory
cwd = r"C:\Users\Administrator\Desktop\Thesis\1. Code"
input_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\test_multivariate__pts_per_period_100__periods_12.json.gz"

dataname = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(cwd, "results", "results_" + dataname + "22.json")
output_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\1. Code\results\results_test_multivariate__pts_per_period_100__periods_12.json22.json"
dataname = dataname.replace("test", "train")
hyperparameter_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\1. Code\hyperparameters\hyperparameters_multivariate_train_multivariate__pts_per_period_100__periods_12.json"



# =======================================================
# 3. GPU setup
# =======================================================
has_gpu = torch.cuda.is_available()
print("has gpu: ", has_gpu, flush=True)
n = torch.cuda.device_count()
print(f"{n} devices found.", flush=True)
if not has_gpu:
    warnings.warn("No GPU found.")
else:
    warnings.warn("GPU working.")



# =======================================================
# 4. Load data and hyperparameters
# =======================================================
import gzip, json

with gzip.open(input_path, "rt") as f:
    equation_data = json.load(f)

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()

print("Looking for:", input_path)
print("Dataset loaded type:", type(equation_data))
print("Keys:", list(equation_data.keys()))
print("Number of attractors:", len(equation_data))

# =======================================================
# 5. Model registry (NeuralForecast only)
# =======================================================
NF_MODELS = {"TiDE": TiDE}



# =======================================================
# Choose which attractors to train
# =======================================================
TRAIN_ALL = False  # ✅ Toggle this to True if you want to train all attractors
TARGET_ATTRACTOR = "Aizawa"  # ✅ Choose the one attractor you want

# Build list of attractors
if TRAIN_ALL:
    attractor_list = list(equation_data.keys())
else:
    if TARGET_ATTRACTOR not in equation_data:
        raise ValueError(f"'{TARGET_ATTRACTOR}' not found! Available attractors: {list(equation_data.keys())}")
    attractor_list = [TARGET_ATTRACTOR]


# =======================================================
# 6. Main training loop
# =======================================================
for equation_name in attractor_list:
    print("\n" + "=" * 80)
    print(f"🔹 Training attractor: {equation_name}")
    print("=" * 80 + "\n", flush=True)
    #print("Checking model:", model_name)
    #print(f"🚀 Now training model '{model_name}' for attractor '{equation_name}'", flush=True)

    train_data = np.copy(np.array(equation_data[equation_name]["values"]))

    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()

    split_point = int(5 / 6 * len(train_data))
    if LONG:
        split_point = int(1 / 6 * len(train_data))  # long horizon

    y_train, y_val = train_data[:split_point], train_data[split_point:]

    all_results[equation_name]["values"] = np.squeeze(y_val)[:-1].tolist()

    for model_name in all_hyperparameters[equation_name].keys():
        print("Checking model:", model_name)
        if (model_name in all_results[equation_name].keys()) and not FORCE_RETRAIN:
            print("Skipping", model_name, "— already trained")
            continue
        # optional: clear old entry
        all_results[equation_name][model_name] = dict()

        print(equation_name + " " + model_name, flush=True)

        pc = dict(all_hyperparameters[equation_name][model_name])


        # =======================================================
        # Create model
        # =======================================================
        if model_name not in NF_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        ModelClass = NF_MODELS[model_name]

        # Forecast horizon = length of validation set
        h = len(y_val)
        pc["h"] = h

        # ✅ Special case for TiDE: must also define input_size safely
        if model_name == "TiDE":
            if "input_size" not in pc:
                # Pick something reasonable but never longer than y_train
                pc["input_size"] = min(max(h * 2, 10), len(y_train) // 3)
                print(f"[TiDE] input_size not in JSON, using default = {pc['input_size']}")
            else:
                # Safety check: cap provided input_size if it's too big
                old_size = pc["input_size"]
                pc["input_size"] = min(pc["input_size"], len(y_train) - 1)
                if pc["input_size"] != old_size:
                    print(f"[TiDE] input_size adjusted from {old_size} to {pc['input_size']} to fit training length")

            # Limit training to 100 steps
            pc["max_steps"] = 500

        # ✅ Filter hyperparameters to remove invalid ones
        sig = inspect.signature(ModelClass.__init__)
        valid_keys = sig.parameters.keys()
        pc_clean = {k: v for k, v in pc.items() if k in valid_keys}

        # Instantiate NeuralForecast with the cleaned hyperparameters
        model = NeuralForecast(models=[ModelClass(**pc_clean)], freq=1)  # freq must exist but arbitrary


        # =======================================================
        # Prepare data for NeuralForecast
        # =======================================================
        if y_train.ndim == 1:
            n_steps, n_dims = len(y_train), 1
        else:
            n_steps, n_dims = y_train.shape

        for d in range(n_dims):
            # Select one series
            y_train_series = y_train[:, d] if n_dims > 1 else y_train

            # Prepare DataFrame for univariate TiDE
            df_train = pd.DataFrame({
                'unique_id': 0,
                'ds': np.arange(len(y_train_series)),
                'y': y_train_series
            })

            # Forecast horizon and training steps
            pc["h"] = len(y_val)
            pc["max_steps"] = 500  # increase from 100 for actual learning

            # Keep only valid hyperparameters
            pc_clean = {k: v for k, v in pc.items() if k in inspect.signature(TiDE.__init__).parameters.keys()}

            # Instantiate NeuralForecast
            model = NeuralForecast(models=[TiDE(**pc_clean)], freq=1)

            # Train
            model.fit(df_train)

            # Predict safely
            fcst = model.predict()
            if model_name in fcst.columns:
                y_val_pred = fcst[model_name].values
            else:
                y_val_pred = fcst.get('y_hat') or fcst.get('y') or np.array([None] * len(y_val))
            
            # Ensure prediction is a list/array and not a pandas Series for consistency
            if hasattr(y_val_pred, 'values'):
                y_val_pred = y_val_pred.values

            # =======================================================
            # Metrics & Save
            # =======================================================
            
            # 1. Get the ground truth specifically for this dimension
            y_val_dim = y_val[:, d] if n_dims > 1 else y_val
            
            # 2. Compute metrics
            try:
                # Ensure we compare arrays of the same length
                min_len = min(len(y_val_dim), len(y_val_pred))
                
                # Prepare inputs: dysts expects (N, D) shapes, reshape if 1D
                y_true_in = y_val_dim[:min_len]
                if y_true_in.ndim == 1:
                    y_true_in = y_true_in.reshape(-1, 1)
                
                y_pred_in = y_val_pred[:min_len]
                if y_pred_in.ndim == 1:
                    y_pred_in = y_pred_in.reshape(-1, 1)

                all_metrics = dysts.metrics.compute_metrics(y_true_in, y_pred_in)
            except Exception as e:
                print(f"Metrics failed for dim {d}: {e}")
                
                # Dummy calculation with reshaped input
                y_dummy = y_val_dim
                if y_dummy.ndim == 1:
                    y_dummy = y_dummy.reshape(-1, 1)
                    
                all_metrics = dysts.metrics.compute_metrics(y_dummy, y_dummy) # dummy
                for key in all_metrics:
                    all_metrics[key] = None

                # 3. Save predictions and metrics
                if n_dims > 1:
                    key = f"{model_name}_dim{d}"
                    all_results[equation_name][key] = {"prediction": y_val_pred.tolist()}
                    all_results[equation_name][key].update(all_metrics)
                else:
                    all_results[equation_name][model_name] = {"prediction": y_val_pred.tolist()}
                    all_results[equation_name][model_name].update(all_metrics)

                # Save incrementally
                with open(output_path, 'w') as f:
                    json.dump(all_results, f, indent=4)

        # =======================================================
        # Train
        # =======================================================
        model.fit(df_train)


        # =======================================================
        # Predict
        # =======================================================
        try:
            fcst = model.predict()
            if model_name in fcst.columns:
                y_val_pred = fcst[model_name].values[:len(y_val)]
            else:
                y_val_pred = fcst['y_hat'].values[:len(y_val)]
        except Exception as e:
            print(f"Failed to predict {equation_name} {model_name}: {e}")
            y_val_pred = np.array([None] * len(y_val))

        all_results[equation_name][model_name]["prediction"] = y_val_pred.tolist()


        # =======================================================
        # Metrics
        # =======================================================
        try:
            y_true_final = y_val
            if y_true_final.ndim == 1:
                y_true_final = y_true_final.reshape(-1, 1)
            
            y_pred_final = y_val_pred
            if y_pred_final.ndim == 1:
                y_pred_final = y_pred_final.reshape(-1, 1)

            all_metrics = dysts.metrics.compute_metrics(y_true_final, y_pred_final)
        except Exception:
            y_dummy = y_val
            if y_dummy.ndim == 1:
                y_dummy = y_dummy.reshape(-1, 1)

            all_metrics = dysts.metrics.compute_metrics(y_dummy, y_dummy)
            for key in all_metrics:
                all_metrics[key] = None

        all_results[equation_name][model_name].update(all_metrics)

        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)

#print("Loaded hyperparameter file:", hyperparameter_path)
#print("Keys in hyperparameter file:", list(all_hyperparameters.keys()))
#print("Available models for Aizawa:", all_hyperparameters["Aizawa"].keys())
#print("Training these attractors:", attractor_list)

# skip possibles points of error
# print models skipped

# skip possibles points of error
# print models skipped
# AAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA