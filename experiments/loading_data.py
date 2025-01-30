import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from competition_code import arc_loader as al
from visualize_grids import plot_task

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

arc_challenge_file = os.path.join(BASE_DIR, 'data', 'arc-agi_training_challenges.json')
# arc_solution_file = os.path.join('data', 'arc-agi_training_solutions.json')

# Load original dataset
dataset = al.ArcDataset.from_file(arc_challenge_file)
print(f"Original dataset size: {dataset.length()}")

# Basic rotation augmentation
rotated = dataset.augment(
    rot=True, # could also be 'rand'
    tp=False, # could also be 'rand'
    shfl_keys=False,
    shfl_ex=False,
    n=1,  # Creates n copies of the augmentations e.g. if rot=T and n=2, then two copies of all three rotations (90, 180, 270) will be created
    seed=42
    )
print(f"Rotated dataset size: {rotated.length()}")

# Plot the original and augmented tasks
task_id = dataset.keys[15]
# plot_task(dataset.queries[task_id], task_id, show_or_save="show")

for key in rotated.keys:
    if key.startswith(task_id):
        plot_task(rotated.queries[key], key, show_or_save="show")


# Reformat data into strings