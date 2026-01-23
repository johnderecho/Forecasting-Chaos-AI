import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import degas as dg

# --- CONFIGURATION ---
DARK = False
FIGPATH = "../private_writing_forecast/fig_resources/"

# List your specific filenames here exactly as they appear in your folder
TIMING_FILES = [
    "./results/timing_standard.json",
    "./results/esn_timing_results.json",
    "./results/nvar_model_times.json"
]

SCORE_FILES = [
    "./results/accuracy_standard.json",
    "./results/esn_scores.json",
    "./results/nvar_final_metrics.json"
]

sns.set_style()
dg.set_style()


# --- UTILITIES ---
def rename_models(ndict):
    """Clean up model names for publication-quality labels."""
    out = {}
    mapping = {
        "NHi TS": "NHiTS", "RNN": "LSTM", "Block RNN": "RNN",
        "XGB": "XGBoost", "ESN": "Echo State", "ExponentialSmoothing": "Exp. Smooth.",
        "Kalman Forecaster": "Kalman", "Linear Regression": "Linear",
        "NODE": "neural ODE", "n ODE": "neural ODE", "TCN": "Temporal ConvNet",
        "NVAR": "nonlin. VAR", "n VAR": "nonlin. VAR"
    }
    for key, value in ndict.items():
        new_key = key.replace("Model", "")
        new_key = re.sub(r'([a-z])([A-Z])', r'\1 \2', new_key)
        new_key = mapping.get(new_key, new_key)
        out[new_key] = value
    return out


def load_from_file_list(file_list, is_timing=False):
    """
    Iterates through an explicit list of files and merges them.
    """
    combined_data = {}

    for fpath in file_list:
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                data = json.load(f)
                for eq_name, models in data.items():
                    if eq_name not in combined_data:
                        combined_data[eq_name] = {}

                    # Cleanup internal keys/metadata
                    if "values" in models: models.pop("values")
                    if "NODE" in models: models["nODE"] = models.pop("NODE")

                    # Merge these models into the specific equation
                    combined_data[eq_name].update(models)
        else:
            print(f"Warning: File not found at {fpath}")

    # Pivot: {Equation: {Model: Data}} -> {Model: {Equation: Value}}
    pivoted = {}
    for eq, models in combined_data.items():
        for model, content in models.items():
            if model not in pivoted: pivoted[model] = {}

            # Extract the specific metric you need
            if is_timing:
                # Adjust key name if your timing JSON uses something other than "Train time"
                pivoted[model][eq] = content.get("Train time", np.nan)
            else:
                # Adjust key name if your score JSON uses "mse", "rmse", or "mae"
                pivoted[model][eq] = content.get("mse", np.nan)

    return rename_models(pivoted)


# --- DATA PROCESSING ---
# Load using the explicit lists
timings = load_from_file_list(TIMING_FILES, is_timing=True)
all_scores_dict = load_from_file_list(SCORE_FILES, is_timing=False)

# Get unified list of models present in BOTH datasets
model_names = sorted(list(set(timings.keys()) & set(all_scores_dict.keys())))
colors = sns.color_palette("husl", len(model_names))
color_dict = dict(zip(model_names, colors))

# --- PLOTTING ---
plt.figure(figsize=(6, 6))
all_all_pairs = []

for model_name in model_names:
    pairs = []
    for eq_name in timings[model_name]:
        if eq_name in all_scores_dict[model_name]:
            t = timings[model_name][eq_name]
            e = all_scores_dict[model_name][eq_name]
            if not np.isnan(t) and not np.isnan(e):
                pairs.append((t, e))

    if not pairs: continue
    pairs = np.array(pairs)

    # Plotting logic (log-transform X axis)
    plot_pairs = pairs.copy()
    plot_pairs[:, 0] = np.log10(plot_pairs[:, 0])

    plt.plot(
        np.median(plot_pairs[:, 0]), np.median(plot_pairs[:, 1]),
        '.', markersize=15, color=color_dict[model_name]
    )

    dg.plot_cross(
        plot_pairs, color=color_dict[model_name],
        center="median", slope="spearman", scale=0.1, aspect=1 / 40
    )
    all_all_pairs.append(plot_pairs)

# Add regression line
if all_all_pairs:
    all_all_pairs = np.concatenate(all_all_pairs)
    dg.plot_linear_confidence(all_all_pairs[:, 0], all_all_pairs[:, 1])

# Formatting
plt.gca().set_xticklabels([f"$10^{{{int(x)}}}$" for x in plt.gca().get_xticks()])
plt.xlabel("Training Time (Log Seconds)")
plt.ylabel("Error (MSE)")
dg.fixed_aspect_ratio(1)
dg.better_savefig(os.path.join(FIGPATH, "timing_vs_error.png"))