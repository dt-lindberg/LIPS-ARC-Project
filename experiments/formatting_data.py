import os
import shutil

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from competition_code import arc_loader as al

# BASE_DIR = r"C:\Users\denna\Downloads\ARC-AGI\ARChitects"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

arc_challenge_file = os.path.join(BASE_DIR, 'data', 'arc-agi_training_challenges.json')

# Load original dataset
dataset = al.ArcDataset.from_file(arc_challenge_file)
print(f"Original dataset size: {dataset.length()}")

# 1. Basic rotation augmentation
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

def download_model(repo_id, store_path, get_name=lambda x: x.split('/')[-1]):
    """Downloads and copies model files instead of using symlinks"""
    model_name = get_name(repo_id)
    model_path = os.path.join(store_path, model_name, 'transformers', 'default', '1')
    
    if not os.path.exists(model_path):
        try:
            # Download model
            download_path = snapshot_download(repo_id=repo_id)
            
            # Create directories
            os.makedirs(os.path.split(model_path)[0], exist_ok=True)
            
            # Copy files instead of symlink
            shutil.copytree(download_path, model_path)
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    
    return model_path

base_model = download_model("da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit", BASE_DIR)
MyFormatter = al.ArcFormatter_premix_3
tokenizer = AutoTokenizer.from_pretrained(base_model)

MyFormatter = MyFormatter(tokenizer=tokenizer)

print("Original Format:")
print(MyFormatter.fmt_train(dataset.queries[task_id]['train']).split("\n"))

print("\nRotated Versions:")
for key in rotated.keys:
    if key.startswith(task_id):
        print(f"\nRotation: {key}")
        print(MyFormatter.fmt_train(rotated.queries[key]['train']).split("\n"))
