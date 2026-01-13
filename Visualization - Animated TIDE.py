# ===================== FORCE INTERACTIVE BACKEND =====================
import matplotlib

matplotlib.use("TkAgg")  # safest backend on Windows

# ===================== IMPORTS =====================
import os
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# ===================== CONFIG =====================
RESULTS_PATH = r"C:\Users\Windows\Desktop\Thesis - GPU\results\MLP TIDE - Lorenz - 50.json"
DATA_PATH = r"C:\Users\Windows\Desktop\Derecho - Thesis\dysts\data\test_multivariate__pts_per_period_100__periods_12.json.gz"
LONG = True
ANIMATION_INTERVAL = 35                                                                                                     # Increased slightly for smoother GIF playback
OUTPUT_DIR = r"C:\Users\Windows\Desktop\Thesis - GPU\animations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== UTILS =====================
def ensure_3cols(array, name):
    array = np.asarray(array)
    if array.ndim == 1:
        if array.size < 3:
            raise ValueError(f"{name} too short for 3D plotting")
        array = array.reshape(-1, 3)
    elif array.ndim == 2 and array.shape[1] < 3:
        last_col = array[:, -1:]
        while array.shape[1] < 3:
            array = np.hstack([array, last_col])
    return array


# ===================== ANIMATION =====================
def animate_phase_space_3d(y_true, y_pred, attractor_name, save_path=None):
    fig = plt.figure(figsize=(10, 8), dpi = 300)
    ax = fig.add_subplot(111, projection="3d")

    # Axis limits
    all_data = np.vstack([y_true, y_pred])
    ax.set_xlim(all_data[:, 0].min(), all_data[:, 1].max())
    ax.set_ylim(all_data[:, 1].min(), all_data[:, 1].max())
    ax.set_zlim(all_data[:, 2].min(), all_data[:, 2].max())

    ax.set_title(f"3D Phase Space Animation: {attractor_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    gt_line, = ax.plot([], [], [], color="black", alpha=0.4,
                       linewidth=1.0, label="Ground Truth")
    pred_line, = ax.plot([], [], [], color="crimson",
                         linestyle="--", linewidth=1.5,
                         label="TiDE Prediction")
    ax.legend()

    def update(frame):
        # Using a stride for smoother/faster GIF if data is very long
        idx = frame
        gt_line.set_data(y_true[:idx, 0], y_true[:idx, 1])
        gt_line.set_3d_properties(y_true[:idx, 2])

        pred_line.set_data(y_pred[:idx, 0], y_pred[:idx, 1])
        pred_line.set_3d_properties(y_pred[:idx, 2])
        return gt_line, pred_line

    # Optimization: GIFs get very large. If y_true is huge, we'll sub-sample frames.
    total_frames = len(y_true)
    max_frames = 300                                                                                                       # Adjust this if you want a longer or shorter GIF
    step = max(1, total_frames // max_frames)
    frames_to_render = range(0, total_frames, step)

    anim = FuncAnimation(
        fig,
        update,
        frames=frames_to_render,
        interval=ANIMATION_INTERVAL,
        blit=False
    )

    # Save as GIF using Pillow
    if save_path:
        # Ensure extension is .gif
        if not save_path.endswith('.gif'):
            save_path = os.path.splitext(save_path)[0] + ".gif"

        print(f"Saving animation to {save_path} using Pillow...")
        try:
            writer = PillowWriter(fps=1000 / ANIMATION_INTERVAL)
            anim.save(save_path, writer=writer)
            print("Saved GIF successfully.")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    plt.show(block=True)
    plt.close(fig)


# ===================== MAIN =====================
def main():
    # Load results
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    # Load ground truth
    with gzip.open(DATA_PATH, "rt") as f:
        data_json = json.load(f)

    attractor_name = "Lorenz"

    if attractor_name not in results:
        print(f"No results found for {attractor_name}")
        return
    if attractor_name not in data_json:
        print(f"No ground truth found for {attractor_name}")
        return

    full_series = np.array(data_json[attractor_name]["values"])
    n_total = len(full_series)

    # Calculate split point based on your LONG flag
    split_point = int(1 / 6 * n_total) if LONG else int(5 / 6 * n_total)
    y_true = full_series[split_point:]

    y_pred = np.array(results[attractor_name]["TiDE_final"]["prediction"])

    if y_pred.size == 0:
        print(f"Empty prediction for {attractor_name}")
        return

    min_len = min(len(y_true), len(y_pred))
    y_true = ensure_3cols(y_true[:min_len], "y_true")
    y_pred = ensure_3cols(y_pred[:min_len], "y_pred")

    # Output path as .gif
    save_path = os.path.join(OUTPUT_DIR, f"TIDE {attractor_name}_animation.gif")

    animate_phase_space_3d(y_true, y_pred, attractor_name=f"{attractor_name} (GT vs TiDE)",
                           save_path=save_path)


if __name__ == "__main__":
    main()