![image](https://raw.githubusercontent.com/da-fr/arc-prize-2024/master/.github/overview.png)

# Run on Snellius

Instructions to set up and run the evaluation pipeline on Snellius.

---

## 1. **Set Up the Environment**

### Step 1: Create a New Directory
From the `$HOME` directory (the default session start location on Snellius), create a new directory and navigate into it:

```bash
mkdir lost_in_program_space
cd lost_in_program_space
```

### Step 2: Clone the Repository
Clone the repository and switch to the appropriate branch:

```bash
git clone -b code-from-dennis https://github.com/nedekriek/MoL-ARC-AGI
cd MoL-ARC-AGI
```

### Step 3: Install Dependencies
Install the required dependencies on a compute node:

```bash
sbatch install_dependencies.sh
```

This command will create two new Conda environments, one for the induction model and one for the transduction model. These environments are automatically activated and deactivated during a run.

---

## 2. **Run the Evaluation Pipeline**

Once the dependencies are installed, execute the evaluation pipeline with:

```bash
sbatch run_evaluation.sh
```

This command will run the pipeline using the following SLURM settings:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
```

---

## 3. **Locate the Results**

After the pipeline finishes running, the results will be saved in a `submission.json` file, located in the following directory:

```
$HOME/lost_in_program_space/MoL-ARC-AGI/output_evaluation_Llama-rearc_with_ttt
```

---

## Files

#### `arc_loader.py`
- **Purpose**: Handles all Data formatting and loading
- **Capabilities**:
   - Class `ArcDataset` which handles all data set related tasks, e.g.:
   - Building datasets from various sources.
   - Modifying, shuffling, and augmenting examples.
   - Splitting, sorting, and filtering examples.
   - Handling dataset keys, challenges and solutions.
   - Preparing the data for tokenization.
   - Creating and verifying submissions.

#### `model_tools.py`
- **Purpose**: Contains code for loading, saving and manipulating models
- **Capabilities**: 
   - Load and Save Model and LoRA adapters
   - Shrink Tokenizer and Embedding Layers
   - Data Collator for masking the task inputs and the first output

#### `inference_tools.py`
- **Purpose**: Contains tools for inference and scoring
- **Capabilities**: 
   - Inference code, including our custom DFS
   - Score calculation

#### `selection.py`
- **Purpose**: Contains functions used to select best answer from different Candidates
- **Capabilities**:
   - Various score aggregation methods
   - Sorting candidates by their score for later submission generation
   - Class `EvalTool` for doing above tasks on-the-fly and printing results

#### `run_finetuning_[model].py`
- **Purpose**: Run the initial finetuning process.
- **Required packages**: `unsloth`
- **Steps**:
   - Load the base model and reduce embedding size.
   - Load and augment training data.
   - Create a lora adapter and execute training.
   - Save the trained lora adapter.
   - Merge the lora model into the base model and save as final model.

#### `run_evaluation_[model].py`
- **Purpose**: Run inference (simuating a kaggle submission).
- **Required packages**: `unsloth` and `diskcache`
- **Steps**:
   - Load the finetuned model.
   - Possibly perform test-time-training on the evaluation set's examples.
   - Save the trained lora adapter for later use.
   - Run inference on the evaluation set.
   - Write a `submission.json` file.
   - Reload and verify the submission file.

## License

Our code is available under the Apache 2.0 license. See the [LICENSE.txt](LICENSE.txt) file for more info.
