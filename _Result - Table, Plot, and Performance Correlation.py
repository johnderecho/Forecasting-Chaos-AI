import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings

# ====== Config ======
DARK = False
FIGPATH = Path(r"C:\Users\Windows\Desktop\Thesis - GPU\Figures")
NOSAVEFIG = False  # your save-figure flag
DG_RED = (0.8, 0.1, 0.1)

FIGPATH.mkdir(parents=True, exist_ok=True)

# ====== Helper Functions ======
def fixed_aspect_ratio(ratio: float) -> None:
    ax = plt.gca()
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    dy = (y_high - y_low)
    dx = (x_right - x_left)
    if dy == 0 or dx == 0:
        return
    ax.set_aspect(abs(dx / dy) * ratio)


def better_savefig(path: Path, dryrun: bool = False) -> None:
    path = Path(path)
    if dryrun:
        print(f"[DRYRUN] Saving figure to: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved figure: {path}")


def make_linear_cmap(colors):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("linear_cmap", colors)


GRANULARITY = 100

BASE_DIR = Path(r"C:\Users\Windows\Desktop\Thesis - GPU")

PATHS = {
    "Lorenz": {
        "BiTCN": BASE_DIR / "Results - All" / "CNN BiTCN.json",
        "TCN": BASE_DIR / "Results - All" / "CNN TCN.json",
        "NBEATS": BASE_DIR / "Results - All" / "MLP NBEATS.json",
        "NHITS": BASE_DIR / "Results - All" / "MLP NHITS.json",
        "TSMixer": BASE_DIR / "Results - All" / "MLP TSMixer.json",
        "SOFTS": BASE_DIR / "Results - All" / "MLP SOFTS.json",
        "MLP Vanilla": BASE_DIR / "Results - All" / "MLP Vanilla.json",
        "DeepNPTS": BASE_DIR / "Results - All" / "MLP DeepNPTS.json",
        "StemGNN": BASE_DIR / "Results - All" / "GNN StemGNN.json",
        "NLinear": BASE_DIR / "Results - All" / "MLP NLinear.json",
        "Vanilla RNN": BASE_DIR / "Results - All" / "RNN Vanilla.json",
        "KAN RMOK": BASE_DIR / "Results - All" / "KAN RMoK.json",

    },
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================
# Summary table (robust to missing models)
# =========================
summary_scores = {}
for dataset_name, models in PATHS.items():
    loaded_models = {}
    for model, path in models.items():
        try:
            loaded_models[model] = load_json(path)
        except FileNotFoundError:
            warnings.warn(f"Missing results file for {dataset_name}/{model}: {path}")
            continue

    if not loaded_models:
        continue

    # Use keys from the first available model
    reference_keys = list(next(iter(loaded_models.values())).keys())

    for key in reference_keys:
        summary_scores.setdefault(key, {})
        for model_name, model_results in loaded_models.items():
            try:
                final_key = next(k for k in model_results[key].keys() if k.endswith("_final"))
                summary_scores[key][model_name] = model_results[key][final_key]["metrics"]["smape"]
            except (KeyError, StopIteration, TypeError):
                continue

summary_df = pd.DataFrame(summary_scores).transpose()

print("\n--- Results Table (SMAPE) ---")
print(summary_df)
print("\nTable Shape:", summary_df.shape)

if not summary_df.empty:
    median_scores = summary_df.median(axis=0, numeric_only=True)
    models_ranked_global = list(median_scores.sort_values().index)
    print("\n--- Models Ranked (Best to Worst) ---")
    print(models_ranked_global)


# ====== Color Palette ======
pastel_rainbow = np.array([
    [0.17254902, 0.45098039, 0.69803922],
    [0.19607843, 0.47843137, 0.5372549],
    [0.372549, 0.596078, 1],
    [0.16862745, 0.20392157, 0.48627451],
    [0.24705882, 0.58039216, 0.42745098],
    [0.65882353, 0.75294118, 0.86666667],
    [0.63921569, 0.85490196, 0.52156863],
    [0.72941176, 0.67843137, 0.83921569],
    [0.53333333, 0.62352941, 0.47843137],
    [0.64705882, 0.70588235, 0.52156863],
    [0.64313725, 0.14117647, 0.48627451],
    [0.92941176, 0.61568627, 0.24705882],
    [0.86666667, 0.23137255, 0.20784314],
    [1.0, 0.3882, 0.2784]
])


# ====== Plot Loop over Attractors ======
for attractor_name, models in PATHS.items():
    # ---- Load results dynamically per attractor ----
    all_scores = {}
    for model_name, path in models.items():
        try:
            results = load_json(path)
        except FileNotFoundError:
            warnings.warn(f"Missing results file for {attractor_name}/{model_name}: {path}")
            continue

        for key in results.keys():
            if key not in all_scores:
                all_scores[key] = {}
            try:
                final_key = next(k for k in results[key].keys() if k.endswith("_final"))
                all_scores[key][model_name] = results[key][final_key]["metrics"]["smape"]
            except (KeyError, StopIteration, TypeError):
                continue

    all_scores = pd.DataFrame(all_scores).transpose()

    if all_scores.empty:
        warnings.warn(f"No plottable scores for {attractor_name}. Skipping.")
        continue

    # ---- Rank models by median ----
    median_scores = all_scores.median(axis=0, numeric_only=True)
    models_ranked = list(median_scores.sort_values().index)

    # ---- Plot Style ----
    if DARK:
        plt.style.use("dark_background")
        fg_color = (1, 1, 1)
        opacity_default = 0.5
    else:
        fg_color = (0, 0, 0)
        opacity_default = 1.0

    palette_models = pastel_rainbow[:len(models_ranked)].tolist()
    palette_models_rev = pastel_rainbow[::-1][:len(models_ranked)].tolist()

    # ---- Violin + Swarm + Point + Box ----
    plt.figure(figsize=(6, 5))  # less cinematic, more statistical

    #sns.violinplot(
    #    data=all_scores,
    #    order=models_ranked,
    #    palette=palette_models_rev,
    #    linewidth=0,
    #    bw_adjust=0.6,
    #    cut=0
    #)

    sns.swarmplot(
        data=all_scores,
        order=models_ranked,
        palette=palette_models,
        size=4,
        alpha=opacity_default
    )

    #sns.pointplot(
    #    data=all_scores,
    #    order=models_ranked,
    #    color=fg_color,
    #    markers=".",
    #    estimator=np.median,
    #    errorbar=("ci", 95),
    #    linestyle="none"
    #)

    #sns.boxplot(
    #    data=all_scores,
    #    order=models_ranked,
    #    color="white",
    #    fliersize=0,
    #    linewidth=0
    #)

    plt.ylim(bottom=0)
    plt.xticks(rotation=45, ha="right")

    # ❌ remove fixed_aspect_ratio entirely

    for spine in plt.gca().spines.values():
        spine.set_alpha(0.5)

    better_savefig(FIGPATH / f"{attractor_name}_model_scores.png", dryrun=NOSAVEFIG)
    plt.close()

    # ---- Model Similarity / Correlation ----
    corr_array = np.array(all_scores.corr(method="spearman"))
    np.fill_diagonal(corr_array, np.nan)
    sort_inds = np.argsort(np.nanmax(corr_array, axis=0))[::-1]
    all_scores_sorted = all_scores.iloc[:, sort_inds]

    #cmap = make_linear_cmap([(1, 1, 1), DG_RED])
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(all_scores_sorted.corr(method="spearman"), annot = True, cmap= "crest", vmin=0, vmax=1, square=True)
    ax.invert_yaxis()
    for item in ax.get_xticklabels():
        item.set_rotation(65)
        item.set_horizontalalignment("right")

    better_savefig(FIGPATH / f"{attractor_name}_model_correlation.png", dryrun=NOSAVEFIG)
    plt.close()

    # ---- Summary Prints ----
    print(f"\nAttractor: {attractor_name}")
    print(all_scores_sorted.isna().sum())
    print(all_scores_sorted.nunique())

vals = all_scores["NBEATS"].dropna()

print("Min:", vals.min())
print("Max:", vals.max())

# Look for the empty region
hist, edges = np.histogram(vals, bins=20)
for h, e1, e2 in zip(hist, edges[:-1], edges[1:]):
    if h == 0:
        print(f"Gap between {e1:.2f} and {e2:.2f}")