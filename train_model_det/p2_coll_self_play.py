import os
import pickle
import numpy as np
import torch
from glob import glob

from src.Board import BOARD_SIZE, DIE_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES


input_dir = "data/self_play_games"
output_file = f"data/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_dataset.pkl"

all_data = []

# Get all .pkl files
files = glob(os.path.join(input_dir, "*.pkl"))

print(f"Found {len(files)} pickle files")
for file in files:
    try:
        with open(file, "rb") as f:
            data = pickle.load(f)

            # Ensure it's iterable (e.g. list of games)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

        print(f"Loaded: {file}")

    except Exception as e:
        print(f"Failed to load {file}: {e}")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save merged data
with open(output_file, "wb") as f:
    pickle.dump(all_data, f)

print(f"\nSaved {len(all_data)} items to {output_file}")
