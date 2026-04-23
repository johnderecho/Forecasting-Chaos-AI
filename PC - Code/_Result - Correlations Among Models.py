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



























