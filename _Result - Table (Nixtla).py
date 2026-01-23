import os
import json
import gzip
import numpy as np
import pandas as pd

GRANULARITY = 100

# Define paths
# path_nbeats = r"C:\Users\Windows\Desktop\Thesis - GPU\results\MLP NBEATS - Lorenz - 50.json"  # Nixtla
path_nbeats = r"C:\Users\Windows\Desktop\Thesis - GPU\original_code_results\result.json"
path_tide = r"C:\Users\Windows\Desktop\Thesis - GPU\results\MLP TIDE - Lorenz - 50.json"
path_tsmixer = r"C:\Users\Windows\Desktop\Thesis - GPU\results\MLP TSMixer - Lorenz - 50.json"

# Load the actual data from the JSON files
def load_json(path):
    with open(path, "r") as file:
        return json.load(file)

all_results_nbeats = load_json(path_nbeats)
all_results_tide = load_json(path_tide)
all_results_tsmixer = load_json(path_tsmixer)

# Using tide results as the primary key reference (previously all_results1)
all_results1 = all_results_tide 
all_results2 = all_results_nbeats

all_scores = dict()

for key in all_results1.keys():
    if key not in all_results2.keys():
        continue
    all_scores[key] = dict()
    
    # Fill scores from the models
    # Note: Adjusting logic to match the structure where 'NBEATS_final' or similar keys exist
    # based on the MLP - NBEATS.py structure provided in context.
    
    try:
        # Extracting SMAPE from the metrics stored in your JSON structure
        # Assuming the structure: all_results[attractor]["NBEATS_final"]["metrics"]["smape"]
        all_scores[key]["nBEATS"] = all_results_nbeats[key]["NBEATS_final"]["metrics"]["smape"]
        all_scores[key]["TiDE"] = all_results_tide[key]["TiDE_final"]["metrics"]["smape"]
        all_scores[key]["TSMixer"] = all_results_tsmixer[key]["TSMixer_final"]["metrics"]["smape"]
    except KeyError:
        # Fallback if the naming convention in JSON differs
        continue

all_scores = pd.DataFrame(all_scores).transpose()
all_scores_dict = all_scores.to_dict()

print("\n--- Results Table (SMAPE) ---")
print(all_scores)
print("\nTable Shape:", all_scores.shape)

print(all_scores.shape)

# find mean and sort to get column names
mean_scores = all_scores.median(axis=0)
sort_order = np.argsort(np.array(mean_scores))
mean_scores = mean_scores.sort_values()
models_ranked = list(mean_scores.index)

print("\n--- Models Ranked (Best to Worst) ---")
print(models_ranked)

#all_scores_dict = rename_models(all_scores_dict)