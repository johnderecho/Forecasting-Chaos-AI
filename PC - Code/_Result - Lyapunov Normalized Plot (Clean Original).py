import gzip
import matplotlib.pyplot as plt
from dysts.metrics import smape, nrmse, spearman, mase, mutual_information
from dysts.metrics import coefficient_of_variation, rmse, r2_score, wape, smape
from dysts.metrics import smape as sp
import dysts.flows as dfl
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np




# ----------------------------
# Helpers
# ----------------------------

def load_file(file_path):
    """
    Load a data file into a pandas DataFrame.

    Supports:
    - CSV files
    - TSV / tab-separated files
    - whitespace-separated text files
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in {".tsv", ".tab"}:
            return pd.read_csv(file_path, sep="\t")
        else:
            # Try common text formats with automatic delimiter detection first
            try:
                return pd.read_csv(file_path, sep=None, engine="python")
            except Exception:
                # Fall back to whitespace-separated data
                return pd.read_csv(file_path, delim_whitespace=True)
    except Exception as e:
        raise ValueError(f"Failed to load '{file_path}': {e}") from e


def load_json(path: str, encoding: str = "utf-8"):
    """Load a plain JSON file from disk."""
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def load_gz_json(path: str, encoding: str = "utf-8"):
    """Load a gzipped JSON file from disk."""
    with gzip.open(path, "rt", encoding=encoding) as f:
        return json.load(f)





# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path(r"C:\Users\Administrator\Desktop\Visualization")
RESULTS_DIR = BASE_DIR / "Results"

PATHS = {}

for file in RESULTS_DIR.glob("*.json"):
    model_name = file.stem  # filename becomes model name
    PATHS[model_name] = file

GROUND_TRUTH_PATH = r"C:\Users\Administrator\Desktop\Visualization\data\test_multivariate__pts_per_period_100__periods_12.json.gz"

# ----------------------------
# LOAD DATA FROM PATHS (FIXED)
# ----------------------------
results = {}

for model_name, path in PATHS.items():

    if not path.exists():
        print(f"Missing file: {path}")
        continue

    data = load_json(str(path))

    # data is expected to contain MANY equations
    for equation_name, eq_block in data.items():

        if equation_name not in results:
            results[equation_name] = {}

        for inner_model_name, inner_block in eq_block.items():
            if isinstance(inner_block, dict) and "prediction" in inner_block:
                results[equation_name][inner_model_name] = inner_block

# print("Loaded equations:", list(results.keys())[:10])
print("Total equations loaded:", len(results))





# ----------------------------
# LOAD DATA
# ----------------------------
ground_truth = load_gz_json(GROUND_TRUTH_PATH)   # contains ["values"]
#results = load_json(RESULTS_PATH)                # contains predictions, usually no ["values"]

# Choose ONE metric for survival traces:
ERROR_METRIC = "smape"  # "wape" or "nrmse"

# print("GROUND TRUTH KEYS:", ground_truth.keys())

if "Lorenz" in ground_truth:
    print("Lorenz structure:", ground_truth["Lorenz"].keys())





# ----------------------------
# Fetch Lyapunov exponents (only for equations that exist in dysts.flows)
# ----------------------------
all_lyaps = {}
for equation_name in results.keys():
    try:
        eq = getattr(dfl, equation_name)()

        # SANITY CHECK: Lyapunov exponent is not None
        lyap = eq.maximum_lyapunov_estimated

        # SANITY CHECK: Lyapunov validity
        if lyap is None or np.isnan(lyap) or np.isinf(lyap):
            print(f"⚠️ Invalid Lyapunov exponent: {equation_name} → {lyap}")
            continue
        if lyap <= 0:
            print(f"⚠️ Non-chaotic or weakly chaotic system: {equation_name} → λ={lyap}")

    except AttributeError:
        print(f"Warning: '{equation_name}' not found in dysts.flows; skipping lyapunov exponent.")
        continue
    all_lyaps[equation_name] = eq.maximum_lyapunov_estimated

if "Lorenz" in all_lyaps:
    print(all_lyaps["Lorenz"])





# ----------------------------
# Survival analysis ("error vs forecast length")
# ----------------------------
all_outputs = {}

for equation_name, eq_results in results.items():
    if equation_name not in ground_truth or "values" not in ground_truth[equation_name]:
        print(f"Skipping '{equation_name}': no ground truth ['values'] in ground_truth file.")
        continue

    true_vals = np.array(ground_truth[equation_name]["values"], dtype=float)

    if not isinstance(eq_results, dict):
        print(f"Skipping '{equation_name}': unexpected results format (expected dict).")
        continue

    all_traces = {}
    for model_name, model_block in eq_results.items():
        # expecting something like results[equation_name][model_name]["prediction"]
        if not isinstance(model_block, dict) or "prediction" not in model_block:
            continue

        pred_vals = np.array(model_block["prediction"])
        if pred_vals.size == 0:
            continue

        # Skip if contains None
        if pred_vals.dtype == object and any(v is None for v in pred_vals.ravel()):
            continue

        # Convert to float if possible
        try:
            pred_vals = pred_vals.astype(float, copy=False)
        except (TypeError, ValueError):
            print(f"Skipping {equation_name}/{model_name}: prediction not numeric.")
            continue

        # Align lengths (common prefix)
        n = min(true_vals.shape[0], pred_vals.shape[0])
        if n < 2:
            continue

        tv = true_vals[:n, :]
        pv = pred_vals[:n, :]

        # SANITY CHECK: shape consistency
        if tv.shape != pv.shape:
            print(f"⚠️ Shape mismatch: {equation_name} - {model_name}: {tv.shape} vs {pv.shape}")
            continue

        all_errs = []
        scale = np.std(tv, axis=0)

        for i in range(1, n):
            true_sub = tv[:i, :]
            pred_sub = pv[:i, :]

            if ERROR_METRIC == "nrmse":
                err_val = nrmse(true_sub, pred_sub, scale=scale)
            elif ERROR_METRIC == "wape":
                err_val = wape(true_sub, pred_sub)
            elif ERROR_METRIC == "smape":
                err_val = smape(true_sub, pred_sub)
            else:
                raise ValueError(f"Unknown ERROR_METRIC: {ERROR_METRIC}")

            all_errs.append(err_val)

        all_traces[model_name] = np.array(all_errs, dtype=float)

    if all_traces:
        all_outputs[equation_name] = all_traces

print(f"Computed survival traces for {len(all_outputs)} equations.")





## save all intermediate outputs using json.dump and gzip
# import json
# from dysts.utils import convert_json_to_gzip
# fname = "./private_archive/all_outputs_long_forecasting.json"
# with open(fname, 'w') as file:
#     all_output_list = dict()
#     for key in all_outputs.keys():
#         all_output_list[key] = dict()
#         for key2 in all_outputs[key].keys():
#             output_vals = all_outputs[key][key2]
#             output_vals[np.isnan(output_vals)] = None
#             all_output_list[key][key2] = list(output_vals)
#     json.dump(all_output_list, file, indent=4, sort_keys=True)
# convert_json_to_gzip(fname)

# ## Load all outputs from cell above
# import json, gzip
# with gzip.open('./private_archive/all_outputs_long_forecasting_smape.json.gz', 'rt', encoding="utf-8") as file:
#    all_outputs = json.load(file)
#    for key in all_outputs.keys():
#        for key2 in all_outputs[key].keys():
#            all_outputs[key][key2] = np.array(all_outputs[key][key2])





# Statistical Aggregation over time
# asks: "On average, across all possible chaotic systems, how does this model's error grow as time passes?"

from scipy.stats import median_abs_deviation
all_smape_series = dict()
for model_name in all_outputs["Aizawa"].keys():
    all_errs = []
    for equation_name in all_outputs.keys():
        if model_name not in all_outputs[equation_name].keys():
            print(f"skipping {model_name} for {equation_name}")
            continue
        err_val = all_outputs[equation_name][model_name]
        all_errs.append(err_val)
    all_errs = np.array(all_errs)

    all_smape_series[model_name] = dict()
    all_smape_series[model_name]["median"] = np.nanmedian(all_errs, axis=0)
    all_smape_series[model_name]["p75"] = np.percentile(all_errs, 75, axis=0)
    all_smape_series[model_name]["p25"] = np.percentile(all_errs, 25, axis=0)
    all_smape_series[model_name]["mad"] = median_abs_deviation(all_errs, axis=0)
    all_smape_series[model_name]["mean"] = np.nanmean(all_errs, axis=0)
    all_smape_series[model_name]["stderr"] = np.nanstd(all_errs, axis=0) / np.sqrt(all_errs.shape[0])





## Plot in Lyapunov time units

import dysts.flows as dfl

max_lyap_times = list()
# units of Lyapunov time



equation_names = sorted(list(all_outputs.keys()))      # already in your code
model_names = list(all_outputs[next(iter(all_outputs))].keys())  # already in your code

# Claude suggestion until timepoint_grid is built
real_max = []
for equation_name in equation_names:
    try:
        eq = getattr(dfl, equation_name)()
        tt, _ = eq.make_trajectory(1000, resample=True, return_times=True, pts_per_period=100)
        dtval = np.median(np.diff(tt))
        lyap = eq.maximum_lyapunov_estimated
        if lyap is None or np.isnan(lyap) or lyap <= 0:
            continue
        n_steps = len(all_outputs[equation_name][next(iter(all_outputs[equation_name]))])
        real_max.append(lyap * n_steps * dtval)
    except Exception:
        continue

actual_max = np.percentile(real_max, 10)
# print(f"Data actually reaches up to (10th percentile): {actual_max:.4f}")

# Step 2: now build the grid to match reality
timepoint_grid = np.logspace(-2, np.log10(actual_max), 500)





## Get the data to lookup timescales
# cwd = os.getcwd()
# input_path = os.path.dirname(cwd)
# input_path += "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"
# equation_data = load_file(input_path)


def interpolate_nan(y):
    """Interpolate over NaNs in a 1D array."""
    x = np.arange(y.shape[0])
    y = y.copy()
    y[np.isnan(y)] = np.interp(x[np.isnan(y)], x[~np.isnan(y)], y[~np.isnan(y)])
    return y


all_prediction_results = dict()
equation_names = sorted(list(all_outputs.keys()))

# model_names = list(all_outputs["Aizawa"].keys())
model_names = list(all_outputs[next(iter(all_outputs))].keys())
for model_name in model_names:

    all_predictions = list()
    for equation_name in equation_names:

        ## Get the times
        # time_vals = equation_data.dataset[equation_name]["time"]
        # dtval = np.median(np.diff(time_vals))
        eq = getattr(dfl, equation_name)()
        tt, _ = eq.make_trajectory(
            1000,
            resample=True,
            return_times=True,
            pts_per_period=100
        )
        dtval = np.median(np.diff(tt))

        ## Get the Lyapunov exponent
        lyap = eq.maximum_lyapunov_estimated

        if model_name not in all_outputs[equation_name].keys():
            print(f"skipping {model_name} for {equation_name}")
            all_predictions.append(np.nan * timepoint_grid)
            continue
        prediction = all_outputs[equation_name][model_name]
        prediction = interpolate_nan(prediction)
        time_vals = np.arange(prediction.shape[0])
        time_vals_lyap = lyap * time_vals * dtval
        max_lyap_times.append(time_vals_lyap[-1])

        ## Account for jaggedness by only resample in valid interpolation range
        cutoff_index = np.where(time_vals_lyap <= timepoint_grid[-1])[0][-1]
        prediction_grid = np.zeros(len(timepoint_grid))
        prediction_resample = np.interp(
            timepoint_grid[:cutoff_index],
            time_vals_lyap,
            prediction,
        )

        prediction_grid[:cutoff_index] += prediction_resample
        prediction_grid[cutoff_index:] = np.nan
        all_predictions.append(prediction_grid)
    all_predictions = np.array(all_predictions)

    all_prediction_results[model_name] = all_predictions
num_dof = np.sum(~np.isnan(all_predictions), axis=0)

## Print 10th and 90th percentile of Lyapunov exponent-scaled times
print(np.percentile(max_lyap_times, 10))
print(np.percentile(max_lyap_times, 50))
print(np.percentile(max_lyap_times, 90))

average_prediction = np.nanmean(all_predictions, axis=0)
error_prediction = np.nanmean(all_predictions, axis=0)


plt.figure()
plt.semilogx(timepoint_grid, average_prediction, color='w')
plt.xlim([None, np.max(timepoint_grid)])

## Confirm that truncation didn't drop statistical power
plt.plot(num_dof)
plt.plot(np.nanstd(all_predictions, axis=0))





## Plot all models mean and errors
bg_color = np.array([1, 1, 1])  # white background
gray_color = np.array([0.7, 0.7, 0.7])

# Auto-assign a distinct color to each model
cmap = plt.colormaps.get_cmap("tab10").resampled(len(model_names))
color_dict = {}
for i, name in enumerate(model_names):
    color_dict[name] = np.array(cmap(i)[:3])  # RGB only, no alpha


plt.figure(figsize=(6.4, 4.8))
model_names = list(all_prediction_results.keys())
all_results_rescaled = []
zorder_index = 0

for model_name in model_names[::-1]:
    results_ave = np.nanmean(all_prediction_results[model_name], axis=0)
    all_results_rescaled.append(results_ave)
    results_std = np.nanstd(all_prediction_results[model_name], axis=0)
    results_dof = np.sum(~np.isnan(all_prediction_results[model_name]), axis=0)
    results_stderr = results_std / np.sqrt(results_dof)

    color = color_dict[model_name]
    ## Plot gray models behind colored models
    if np.all(color == gray_color):
        zorder_shift = -100
    else:
        zorder_shift = 0

    plt.fill_between(
        timepoint_grid,
        results_ave - results_stderr,
        results_ave + results_stderr,
        color=bg_color,
        alpha=0.5,
        zorder=zorder_index + zorder_shift - 1
    )
    plt.fill_between(
        timepoint_grid,
        results_ave - results_stderr,
        results_ave + results_stderr,
        color=color_dict[model_name],
        alpha=0.2,
        zorder=zorder_index + zorder_shift
    )
    plt.semilogx(
        timepoint_grid,
        results_ave,
        linewidth=2,
        color=color_dict[model_name],
        zorder=zorder_index + 1 + zorder_shift
    )
    zorder_index += 2
all_results_rescaled = np.array(all_results_rescaled)
#plt.xlim([0.01, 100])
plt.xlim([0.01, actual_max])

## SMAPE
plt.ylim([0, None])
plt.tight_layout()

## WAPE
# plt.yscale('log')
# plt.ylim([1e0, 1e5])
# dg.fixed_aspect_ratio(1/1.5, semilogx=True, semilogy=True)

## MASE
# plt.ylim([0, 40])
# dg.fixed_aspect_ratio(1/1.5, semilogx=True)

## R^2
# plt.ylim([0, 1])
# dg.fixed_aspect_ratio(1/1.5, semilogx=True)

## Spearman
# plt.ylim([None, 1])
# dg.fixed_aspect_ratio(1/1.5, semilogx=True)

## NRMSE
# plt.yscale('log')
# plt.ylim([1e-0, 1e3])
# dg.fixed_aspect_ratio(1/1.5, semilogx=True, semilogy=True)

# plt.ylim([0, 1e6])
## log scale y axis
# plt.ylim([1e-3, 1e0])
# plt.yscale('log')
# dg.fixed_aspect_ratio(1/1.5, semilogx=True, semilogy=True)
# plt.ylim([1e-3, 1e5])
# plt.yscale('log')
# dg.fixed_aspect_ratio(1/1.5, semilogy=True, semilogx=True)

# dg.better_savefig(FIGPATH + "forecast_lengths2.png", dpi=450, dryrun=False)

## make a separate legend for the fill_between
plt.figure()
for i, model_name in enumerate(model_names):
    plt.plot([], color=color_dict[model_name], label=model_name)
plt.legend()
plt.show()

# dg.better_savefig(FIGPATH + "forecast_lengths_legend2.png", dpi=450, dryrun=NOSAVEFIG)










import pandas as pd

all_modelwise_corrs = {}
for model_name in model_names:
    all_modelwise_corrs[model_name] = {}
    all_modelwise_corrs[model_name]["correlations"] = []
    all_modelwise_corrs[model_name]["errors"] = []
    all_modelwise_corrs[model_name]["errors_up"] = []
    all_modelwise_corrs[model_name]["errors_down"] = []

for i in range(len(all_prediction_results['NBEATS'][0])):
    all_vals = []
    for model_name in model_names:
        model_vals = all_prediction_results[model_name][:, i]
        all_vals.append(model_vals)
    all_vals = np.array(all_vals)

    ## compute correlation matrix robust
    # corr_mat = np.corrcoef(all_vals)
    corr_mat = pd.DataFrame(all_vals.T).corr(method="spearman").values
    # hollow out diagonal
    np.fill_diagonal(corr_mat, 0)
    mutual_corr_strength = np.nanmean(corr_mat, axis=0)

    for j, model_name in enumerate(model_names):
        corr = mutual_corr_strength[j]
        all_modelwise_corrs[model_name]["correlations"].append(corr)

        ## error bar with fisher z transform
        corr_err_up = np.tanh(np.arctanh(corr) + 0.674 / np.sqrt(all_vals.shape[1] - 3))
        corr_err_down = np.tanh(np.arctanh(corr) - 0.674 / np.sqrt(all_vals.shape[1] - 3))
        corr_err = np.abs(corr_err_up - corr_err_down) / 2
        all_modelwise_corrs[model_name]["errors_up"].append(corr_err_up)
        all_modelwise_corrs[model_name]["errors_down"].append(corr_err_down)
        all_modelwise_corrs[model_name]["errors"].append(corr_err)





plt.figure(figsize=(4, 4))
for ii, model_name in enumerate(model_names[plot_inds[::-1]]):
    # plt.semilogx(
    #     timepoint_grid,
    #     all_modelwise_corrs[model_name]["correlations"],
    #     label=model_name,
    #     color=color_dict[model_name],
    #     linewidth=1.5
    # )

    # Error bars smaller than 0.01 are not visible
    plt.semilogx([], [])
    vals = np.array(all_modelwise_corrs[model_name]["correlations"])
    errs = np.array(all_modelwise_corrs[model_name]["errors"])
    plt.fill_between(
        timepoint_grid,
        vals - errs,
        vals + errs,
        color=bg_color,
        alpha=0.05,
        zorder=ii - 1
    )
    dg.plot_err(
        vals,
        errs,
        x=timepoint_grid,
        linewidth=2,
        alpha=0.1,
        color=color_dict[model_name],
        label=model_name,
        zorder=ii
    )

plt.xlim([0.01, 100])

# dg.better_savefig(FIGPATH + "mutual_correlation2.png", dpi=800, dryrun=False)





from scipy.stats import spearmanr

all_modelwise_corrs = {}
num_tpts = len(all_prediction_results['NBEATS'][0])
for model_name in model_names:

    all_modelwise_corrs[model_name] = {}
    all_modelwise_corrs[model_name]["correlations"] = []
    all_modelwise_corrs[model_name]["errors"] = []

    all_values = []
    for i in range(num_tpts):
        ## Errors for all systems for that model at that timepoint
        model_vals = all_prediction_results[model_name][:, i]
        ## linalg norm with nans
        norm_val = np.linalg.norm(model_vals[~np.isnan(model_vals)])
        model_vals = model_vals / norm_val
        all_values.append(model_vals)

    # median correlation of all model errors across all timepoints
    base_vals = np.nanmedian(np.array(all_values), axis=0)  # shape (num_systems)

    for i in range(num_tpts):
        model_vals = all_prediction_results[model_name][:, i]
        corr, pval = spearmanr(base_vals, model_vals, nan_policy='omit')

        all_modelwise_corrs[model_name]["correlations"].append(corr)

        ## error bar with fisher z transform
        corr_err_up = np.tanh(np.arctanh(corr) + 0.674 / np.sqrt(len(base_vals) - 3))
        corr_err_down = np.tanh(np.arctanh(corr) - 0.674 / np.sqrt(len(base_vals) - 3))
        corr_err = np.abs(corr_err_up - corr_err_down) / 2
        all_modelwise_corrs[model_name]["errors"].append(corr_err)

        ## Compute error in with bootstrap (slow, errors are smaller
        ## than the more conservative Fisher transform)
        # all_corrs = []
        # for _ in range(500):
        #     idx = np.random.choice(len(model_vals), size=len(model_vals), replace=True)
        #     corr, pval = spearmanr(base_vals[idx], model_vals[idx], nan_policy='omit')
        #     all_corrs.append(corr)
        # corr = np.nanmean(all_corrs)
        # corr_err = np.nanstd(all_corrs) / np.sqrt(len(all_corrs))
        # all_modelwise_corrs[model_name]["correlations"].append(corr)
        # all_modelwise_corrs[model_name]["errors"].append(corr_err)





plt.figure(figsize=(4, 4))
for model_name in model_names[plot_inds[::-1]]:
# plt.semilogx(
#     timepoint_grid,
#     all_modelwise_corrs[model_name]["correlations"],
#     label=model_name,
#     color=color_dict[model_name],
#     linewidth=1.5
# )

# Error bars smaller than 0.01 are not visible
    plt.semilogx([], [])
dg.plot_err(
all_modelwise_corrs[model_name]["correlations"],
np.array(all_modelwise_corrs[model_name]["errors"]),
x = timepoint_grid,
linewidth = 1.5,
alpha = 0.1,
color = color_dict[model_name],
label = model_name,
)

plt.xlim([0.01, 100])

dg.better_savefig(FIGPATH + "mutual_correlation.png", dpi=800, dryrun=False)





## Variance
all_modelwise_vars = {}
num_tpts = len(all_prediction_results['NBEATS'][0])
for model_name in model_names:
    all_modelwise_vars[model_name] = []
for i in range(num_tpts):
    model_vals = all_prediction_results[model_name][:, i]
all_modelwise_vars[model_name].append(np.nanvar(model_vals))

plt.figure(figsize=(4, 4))
for model_name in model_names[sort_inds[::-1]]:
    plt.semilogx(
timepoint_grid,
all_modelwise_vars[model_name],
label = model_name,
color = color_dict[model_name],
linewidth = 2
)
plt.xlim([timepoint_grid[0], timepoint_grid[-1]])

# NOSAVEFIG = Fa
dg.better_savefig(FIGPATH + "variance_across_systems.png", dpi=800, dryrun=True)

## Correlate variance and correlation
plt.figure(figsize=(4, 4))
for model_name in model_names[sort_inds[::-1]]:
    plt.plot(
all_modelwise_corrs[model_name]["correlations"],
all_modelwise_vars[model_name],
'.',
label = model_name,
color = color_dict[model_name]
)
plt.xlabel("Correlation")
plt.ylabel("Variance")

dg.better_savefig(FIGPATH + "correlation_vs_variance.png", dpi=800, dryrun=True)





## Correlation with Lyapunov time
import dysts.flows as dfl

all_lyap_values = list()
for equation_name in equation_names:
    eq = getattr(dfl, equation_name)()
## Get the Lyapunov exponent and convert to timescale
lyap = 1 / eq.maximum_lyapunov_estimated
all_lyap_values.append(lyap)
all_lyap_values = np.array(all_lyap_values)

from scipy.stats import spearmanr

all_modelwise_corrs = {}
num_tpts = len(all_prediction_results['NBEATS'][0])
for model_name in model_names:
    all_modelwise_corrs[model_name] = {}
all_modelwise_corrs[model_name]["correlations"] = []
all_modelwise_corrs[model_name]["errors"] = []

for i in range(num_tpts):
    model_vals = all_prediction_results[model_name][:, i]
corr, pval = spearmanr(all_lyap_values, model_vals, nan_policy='omit')

all_modelwise_corrs[model_name]["correlations"].append(corr)

## error bar with fisher z transform
corr_err_up = np.tanh(np.arctanh(corr) + 0.674 / np.sqrt(len(all_lyap_values) - 3))
corr_err_down = np.tanh(np.arctanh(corr) - 0.674 / np.sqrt(len(all_lyap_values) - 3))
corr_err = np.abs(corr_err_up - corr_err_down) / 2
all_modelwise_corrs[model_name]["errors"].append(corr_err)





plt.figure(figsize=(4, 4))
for model_name in model_names[plot_inds[::-1]]:
# Error bars smaller than 0.01 are not visible
# plt.semilogx([], [])
# dg.plot_err(
#     all_modelwise_corrs[model_name]["correlations"],
#     np.array(all_modelwise_corrs[model_name]["errors"]),
#     x = timepoint_grid,
#     linewidth=1.5,
#     alpha=0.1,
#     color=color_dict[model_name],
#     label=model_name,
# )

    plt.semilogx([], [])
vals = np.array(all_modelwise_corrs[model_name]["correlations"])
errs = np.array(all_modelwise_corrs[model_name]["errors"])
plt.fill_between(
    timepoint_grid,
    vals - errs,
    vals + errs,
    color=bg_color,
    alpha=0.05,
    zorder=ii - 1
)
dg.plot_err(
    vals,
    errs,
    x=timepoint_grid,
    linewidth=2,
    alpha=0.05,
    color=color_dict[model_name],
    label=model_name,
    zorder=ii
)

plt.xlim([0.01, 100])

dg.better_savefig(FIGPATH + "lyapunov_correlation.png", dpi=800, dryrun=False)