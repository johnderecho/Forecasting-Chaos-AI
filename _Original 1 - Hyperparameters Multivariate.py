#!/usr/bin/python

import torch

# Removed the failing import: from dysts_data.datasets import load_file

has_gpu = torch.cuda.is_available()
print("has gpu1: ", torch.cuda.is_available(), flush=True)

import sys
import os
import json
import warnings
import gzip  # Added for compressed files
import numpy as np
import pandas as pd

import dysts
import darts
from darts.models import *
from darts import TimeSeries
import darts.models

cwd = os.path.dirname(os.path.realpath(__file__))

input_path = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\train_multivariate__pts_per_period_100__periods_12.json.gz"
pts_per_period = 100
network_inputs = [5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period]

SKIP_EXISTING = True
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE,
                 darts.utils.utils.SeasonalityMode.NONE,
                 darts.utils.utils.SeasonalityMode.MULTIPLICATIVE]
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE,
                 darts.utils.utils.SeasonalityMode.NONE
                 ]
time_delays = [3, 5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period,
               int(1.5 * pts_per_period)]
time_delays = [3, 5, 10, int(0.25 * pts_per_period)]
network_outputs = [1, 4]
network_outputs = [1]

has_gpu = torch.cuda.is_available()
print("has gpu: ", torch.cuda.is_available(), flush=True)
n = torch.cuda.device_count()
print(f"{n} devices found.", flush=True)
if not has_gpu:
    warnings.warn("No GPU found.")
    gpu_params = {
        "accelerator": "cpu",
    }
else:
    warnings.warn("GPU working.")
    gpu_params = {
        "accelerator": "gpu",
        "devices": n,
    }

pl_trainer_kwargs = [gpu_params]
model_static_dict = {"pl_trainer_kwargs": pl_trainer_kwargs}

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
# output_path = cwd + "/hyperparameters/hyperparameters_multivariate_" + dataname + ".json"
output_path = r"C:\Users\Windows\Desktop\Thesis - GPU\original_code_results\hyper.json"



# --- INTEGRATED MODERN LOADING WAY ---
try:
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        raw_json_data = json.load(f)
except (gzip.BadGzipFile, OSError):
    # Fallback: If the file is NOT actually gzipped, load as normal text
    with open(input_path, "r", encoding="utf-8") as f:
        raw_json_data = json.load(f)

class DystsDataWrapper:
    def __init__(self, data):
        self.dataset = data

equation_data = DystsDataWrapper(raw_json_data)



# -------------------------------------

try:
    with open(output_path, "r") as file:
        all_hyperparameters = json.load(file)
except FileNotFoundError:
    all_hyperparameters = dict()

parameter_candidates = dict()

parameter_candidates["NBEATSModel"] = {
    "input_chunk_length": network_inputs,
    "output_chunk_length": network_outputs,
    "n_epochs": [200],
    "num_stacks": [10],
    "num_blocks": [1],
    "num_layers": [4],
    "layer_widths": [256]
}

for equation_name in equation_data.dataset:
    if equation_name != "Lorenz":
        continue

    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    if equation_name not in all_hyperparameters.keys():
        all_hyperparameters[equation_name] = dict()

    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

    for model_name in parameter_candidates.keys():

        print(equation_name + " " + model_name)
        if SKIP_EXISTING and model_name in all_hyperparameters[equation_name].keys():
            print(f"Entry for {equation_name} - {model_name} found, skipping it.")
            continue
        pc = dict()
        pc.update(parameter_candidates[model_name])
        try:
            pc.update({"pl_trainer_kwargs": [gpu_params]})
            model = getattr(darts.models, model_name)
            model_best = model.gridsearch(pc, y_train_ts, val_series=y_test_ts,
                                          metric=darts.metrics.smape)
        except:
            pc = dict()
            pc.update(parameter_candidates[model_name])
            model = getattr(darts.models, model_name)
            model_best = model.gridsearch(pc, y_train_ts, val_series=y_test_ts,
                                          metric=darts.metrics.smape)

        best_hyperparameters = model_best[1].copy()

        for hyperparameter_name in best_hyperparameters:
            if "season" in hyperparameter_name:
                best_hyperparameters[hyperparameter_name] = best_hyperparameters[hyperparameter_name].name

        all_hyperparameters[equation_name][model_name] = best_hyperparameters

    with open(output_path, 'w') as f:
        json.dump(all_hyperparameters, f, indent=4)
