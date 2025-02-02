![image](https://raw.githubusercontent.com/da-fr/arc-prize-2024/master/.github/overview.png)

# The ARC project for group Lost in Program Space

This is the repo for the submisssion of the arc project for group Lost in Program Space, in the January project [Program induction and the ARC challenge](https://msclogic.illc.uva.nl/current-students/courses/projects/project/231/1st-Semester-2024-25-Program-induction-and-the-ARC-challenge) supervised by [Dr. Fausto Carss](https://faustocarcassi.com/). Our architecture breaks down to the induction part and the transduction part. We eventually ran our program on the first hundred problems in the evaluation problems, due to the test submission being closed for 2024. The final results is as the following:

|| Tasks solved      | Applied on | Accuracy |
| ----------- | ----------- | ------- | ------ |
| Induction model      | 2       | 9 | 77.8% |
| Transduction model   | 49.5      | 91 | 54.4% |
| Combined model | 56.5 | 100 | 56.5% |


The LIPS_reports.pdf is our technical report, please check it out for discussion of our result.

---------
# Acknowledgemet
Our induction model
is adapted from model from the model 🤗[barc0/Llama-3.1-ARC-Potpourri-Induction-8B](https://huggingface.co/barc0/Llama-3.1-ARC-Potpourri-Induction-8B) developed by 
W.-D. Li, K. Hu, C. Larsen, Y. Wu, S. Alford, C. Woo, S. M. Dunn, H. Tang, M. Naim,
D. Nguyen, W.-L. Zheng, Z. Tavares, Y. Pu, and K. Ellis, “Combining Induction and
Transduction for Abstract Reasoning” Dec. 2024. arXiv:2411.02272.

Our transduction model and architecture is adapted from the model model [wb55L_nemomini_fulleval](https://www.kaggle.com/models/dfranzen/wb55l_nemomini_fulleval/Transformers/default/1) given by D. Franzen, J. Disselhoff, and D. Hartmann, “The LLM ARChitect: Solving ARC-
AGI Is A Matter of Perspective”.

-----------

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
git clone https://github.com/dt-lindberg/LIPS-ARC-Project.git
cd LIPS-ARC-Project
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
$HOME/lost_in_program_space/LIPS-ARC-Project/output_evaluation_Llama-rearc_with_ttt
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
