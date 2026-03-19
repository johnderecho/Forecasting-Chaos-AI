#

# Asks how fast (efficient the model is)
# calculates computational time (wall clock)



## Timing results

## Load the timing results
fpath = "./results/results_test_multivariate__pts_per_period_100__periods_12.json_timing.json"
with open(fpath, "r") as file:
    all_timings = json.load(file)
print(len(all_timings.keys()))

## Load additional timing results
fpath = "./results/results_test_multivariate__pts_per_period_100__periods_12.json_timing_esn.json"
with open(fpath, "r") as file:
    all_timings_esn = json.load(file)


## Load additional timing results
fpath = "./results/results_test_multivariate__pts_per_period_100__periods_12.json_timing_nvar.json"
with open(fpath, "r") as file:
    all_timings_nvar = json.load(file)

## Drop unneeded keys
for item in all_timings.keys():
    # delete the "values" key if it exists
    if "values" in all_timings[item].keys():
        all_timings[item].pop("values")

## Merge the timing results
for key in all_timings_esn:
    all_timings[key].update(all_timings_esn[key])
for key in all_timings_nvar:
    all_timings[key].update(all_timings_nvar[key])

## drop any incomplete entry
keys = list(all_timings.keys())
for key in keys:
    if len(all_timings[key]) != len(all_timings["Aizawa"]):
        print(key)
        all_timings.pop(key)

## Rename NODE to nODE
for key in all_timings.keys():
    all_timings[key]["nODE"] = all_timings[key].pop("NODE")

## invert inner and outer keys
timings = dict()
for model_name in all_timings["Aizawa"].keys():
    timings[model_name] = dict()
    for equation_name in all_timings:
        timings[model_name][equation_name] = all_timings[equation_name][model_name]

print(len(timings["RNNModel"]))
timings = rename_models(timings)





import degas as dg

from scipy.stats import spearmanr
all_mean_timings = dict()
all_all_pairs = list()
all_centers = list()
plt.figure(figsize=(4, 4))
for i, model_name in enumerate(model_names):
    if model_name not in timings:
        print(f"Skipping {model_name}")
        continue

    all_pairs = list()
    for equation_name in timings[model_name]:
        try:
            time_val = float(timings[model_name][equation_name]["Train time"])
            # time_val = float(timings[model_name][equation_name]["Inference time"])
            error_val = float(all_scores_dict[model_name][equation_name])
            all_pairs.append((time_val, error_val))
        except KeyError:
            print(f"Skipping {model_name} {equation_name}")
            continue
    all_pairs = np.array(all_pairs)
    mean_time = np.median(all_pairs[:, 0])
    all_mean_timings[model_name] = mean_time
    all_pairs[:, 0] = np.log10(all_pairs[:, 0])
    # all_pairs = all_pairs[::, ::-1]
    # plt.figure()
    # plt.plot(
    # all_pairs[:, 0], all_pairs[:, 1], '.',
    # markersize=2, alpha=0.1, color=pastel_rainbow[i])
    # plt.plot(
    #     all_pairs[:, 0],
    #     all_pairs[:, 1],
    #     '.',
    #     alpha=0.1,
    #     markersize=2,
    #     color=color_dict[model_name],
    #     zorder=-2
    # )
    # dg.plot_linear_confidence(
    #     all_pairs[:, 0],
    #     all_pairs[:, 1],
    #     color=color_dict[model_name],
    #     ci_kwargs = {"color": color_dict[model_name]},
    #     show_pi=False,
    # )

    plt.plot(
        np.median(all_pairs[:, 0]),
        np.median(all_pairs[:, 1]),
        '.',
        markersize=15,
        color=color_dict[model_name],
    )
    # dg.draw_ellipse(all_pairs, fill=True, edgecolor=None, alpha=0.5, facecolor=pastel_rainbow[i])
    dg.plot_cross(
        all_pairs,
        color=color_dict[model_name],
        center="median",
        slope="spearman",
        scale=0.1,
        flip=False,
        aspect=1/40,
    )
    ang = np.arctan(spearmanr(all_pairs[:, 0], all_pairs[:, 1])[0])

    title_str = f"{model_name} (r={spearmanr(all_pairs[:, 1], all_pairs[:, 0])[0]:.2f})"
    print(title_str)
    # plt.title(title_str)

    all_all_pairs.append(all_pairs)
    all_centers.append(np.median(all_pairs, axis=0))

all_all_pairs = np.concatenate(all_all_pairs)
all_centers = np.array(all_centers)

dg.plot_linear_confidence(
    all_all_pairs[:, 0],
    all_all_pairs[:, 1],
    color=(1.0, 1.0, 1.0),
    linewidth=3,
    ci_kwargs = {"color": (0.7, 0.7, 0.7), "zorder":-110},
    show_pi=False,
    ci_range=0.999999,
    zorder=-100
)

from scipy.stats import spearmanr
print(f"Spearman coefficient: {spearmanr(all_all_pairs[:, 0], all_all_pairs[:, 1])[0]}")

# plt.plot(all_all_pairs[:, 0],
#          all_all_pairs[:, 1],
#          '.',
#          markersize=0.5, alpha=0.14, color=fg_color, zorder=-1
# )
# ax = sns.kdeplot(
#     x=all_all_pairs[:, 0],
#     y=all_all_pairs[:, 1],
#     fill=True,
#     thresh=0.1,
#     color=fg_color,
#     shade=False,
#     bw_adjust=1.2,
#     levels=18,
#     alpha=0.1,
#     linewidths=0.5,
#     zorder=-1
# )

min_x, min_y = np.inf, np.inf
max_x, max_y = 0, 0
for item in plt.gca().get_lines():
    vals = item.get_xdata()
    min_x = min(min_x, np.min(vals))
    max_x = max(max_x, np.max(vals))
    vals = item.get_ydata()
    min_y = min(min_y, np.min(vals))
    max_y = max(max_y, np.max(vals))
plt.xlim([min_x, max_x])
plt.ylim([min_y, max_y*1.05])
# print(min_x, max_x, min_y, max_y)
# plt.xlim([25, 130])

## Sort model names by mean time
all_mean_timings = pd.Series(all_mean_timings)
all_mean_timings = all_mean_timings.sort_values()
all_mean_timings = list(all_mean_timings.index)[::-1]
print(all_mean_timings)

# plt.xlim([0, None])
# rename y axis labels to logarithmic
# set in Helvetica
plt.gca().set_xticklabels(
    [f"$10^{{{int(y)}}}$" for y in plt.gca().get_xticks()],
    fontdict={
    "family": 'sans-serif',
    "fontname": "Helvetica",
    }
)

## set aspect ratio of plot to perfect square
dg.fixed_aspect_ratio(1)

# dg.better_savefig(FIGPATH + "timing_vs_error_inference.png")
dg.better_savefig(FIGPATH + "timing_vs_error_training.png")

plt.figure(figsize=(4, 4))
for model_name in model_names:
    plt.plot([], color=color_dict[model_name])
# plt.legend()
# legend with thick lines
# get line2d

from matplotlib.lines import Line2D
plt.legend(
    handles=[
        Line2D([0], [0], color=color_dict[model_name], lw=4, label=model_name)
        for i, model_name in enumerate(model_names)
    ],
    loc="upper left",
    bbox_to_anchor=(1, 1),
    frameon=False,
    fontsize=12,
    title="Model",
    title_fontsize=14,
)

dg.better_savefig(FIGPATH + "timing_vs_error_legend.png", dryrun=NOSAVEFIG)




# How models handle chaotic systems

# side-by-side visual comparisons of how the top-performing models handle a specific chaotic system



equation_name = "MackeyGlass"


from dysts.datasets import load_file
cwd = os.getcwd()
input_path = os.path.dirname(cwd) + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"
equation_data = load_file(input_path)

input_path = os.getcwd() + "/results/results_test_multivariate__pts_per_period_100__periods_12.json.gz"
all_results = load_file(input_path)

prev_vals = np.array(equation_data.dataset[equation_name]["values"])
split_point = int(5/6 * len(prev_vals))
prev_vals = prev_vals[:split_point]

prev_vals = prev_vals[-400:]

true_vals = np.array(all_results[equation_name]["values"])

plt_models = models_ranked[:4]

plt.figure()
plt.plot(
    np.vstack([prev_vals, true_vals])[:, 0],
    linewidth=4,
    zorder=-30,
    color=list(fg_color) + [0.2]
)
for i, model_name in enumerate(plt_models):
    plt.plot(len(prev_vals) + np.arange(len(true_vals) + 1),
        np.array(all_results[equation_name][model_name]["prediction"])[:, 0],
        color=dg.pastel_rainbow[i],
        zorder=-i
    )
dg.vanish_axes()
dg.fixed_aspect_ratio(1/6)

dg.better_savefig(FIGPATH + f"forecast_examples_univariate_{equation_name}.png", dpi=800)

plt.figure()
plt.plot(
        np.vstack([prev_vals, true_vals])[:, 0],
        np.vstack([prev_vals, true_vals])[:, 1],
        linewidth=4,
        zorder=-30,
        color=list(fg_color) + [0.2]
)
for i, model_name in enumerate(plt_models):
    plt.plot(
        np.array(all_results[equation_name][model_name]["prediction"])[:, 0],
        np.array(all_results[equation_name][model_name]["prediction"])[:, 2],
        color=dg.pastel_rainbow[i],
        zorder=-i
    )
# dg.vanish_axes()
# dg.fixed_aspect_ratio(1/6)


dg.better_savefig(FIGPATH + f"forecast_examples_multivariate_{equation_name}.png", dpi=800)