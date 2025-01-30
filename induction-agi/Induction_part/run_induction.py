# Import the basic packages
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import datetime
from tqdm import tqdm
import json
import torch
import numpy as np
import os
import argparse
import pathlib


##################################
# Import the inductor and prompt_creator from the previous snippets.
##################################

from prompt_creator import prompt_creator
from dataloader import load_data
from problem_class import problem
from execute import multi_execute_transformation, parse_code 
from common import *
from validation import compare_grids, multi_validate, validate
from arc import train_problems, validation_problems
from program_debugging import debug_program


##################################
# Configs
##################################

# Load the configuration file
with open('Induction_part/config.json', 'r') as config_file:
    config = json.load(config_file)

num_of_samples_per_problem = config["num_of_samples_per_problem"]
# Here the num_of_samples_per_problem is the total number of program samples we want for a simple problem.
BATCH_SIZE = config["BATCH_SIZE"]
# The number of programs generated per input prompt.

BASE_MODEL = config["BASE_MODEL"]
LORA_DIR = config["LORA_DIR"]
TENSOR_PARALLEL = config["TENSOR_PARALLEL"]
mode = config["MODE"]
prompt_engineering = config["prompt_engineering"]
TEMPERATURE = config["TEMPERATURE"]
TENSOR_PARALLEL = 1
folder_path = config["folder_path"]
MULTI_EXECUTE = config["MULTI_EXECUTE"]
debug_threshold = config["debug_threshold"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
saving_file = config["saving_file"]


##################################
# Load the model and tokenizer
##################################

if LORA_DIR:
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=256, max_model_len=12000,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL)
    lora_request=LoRARequest("barc_adapter", 1, LORA_DIR)
else:
    llm = LLM(model=BASE_MODEL, enable_lora=False, max_model_len=12000,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

##################################
# Load the data
##################################
problem_file = "./arc_problems_validation_400_extra_newline_v2.jsonl"

if mode == "train":
    folder_path = folder_path + "/" + "training"
elif mode == "evaluation":
    folder_path = folder_path + "/" + "evaluation"

problems = load_data(folder_path, mode)

##################################
# Inference
##################################

def get_arc_problem(uid):
    for problem in train_problems + validation_problems:
        if problem.uid == uid:
            return problem
    assert False, f"Problem {uid} not found"

def get_LLM_output():
    prompt_creator_obj = prompt_creator()
    
    
    all_responses = []
    for problem in tqdm(problems):
        messages = prompt_creator_obj.prompt_to_message(prompt_creator_obj.create_prompt_from_test_task(problem))

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        inputs = tokenizer.apply_chat_template([
            {"role":"system", "content":messages[0]["content"]},
            {"role":"user", "content":messages[1]["content"]}
        ], tokenize=False, add_generation_prompt=True)
        input_tokens = tokenizer.apply_chat_template([
            {"role":"system", "content":messages[0]["content"]},
            {"role":"user", "content":messages[1]["content"]}
        ], tokenize=True, add_generation_prompt=True)
        print(f"Number of tokens: {len(input_tokens)}")
        if len(input_tokens) > 8000:
            print("skip!!!!!")
            continue

        assert num_of_samples_per_problem % BATCH_SIZE == 0
        if  len(input_tokens) < 1750:
            tmp_batch_size = BATCH_SIZE * 4
        elif len(input_tokens) < 4000:
            # double the number of samples
            tmp_batch_size = BATCH_SIZE * 4
        elif len(input_tokens) < 5000:
            tmp_batch_size = BATCH_SIZE 
        else:
            tmp_batch_size = BATCH_SIZE

        print(f"batch size: {tmp_batch_size}")
        sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=1536,
                                        n=tmp_batch_size)
        aggregate_outputs = []
        for i in range(num_of_samples_per_problem // tmp_batch_size):
            if LORA_DIR:
                outputs = llm.generate(
                    inputs,
                    sampling_params,
                    lora_request=lora_request
                )
            else:
                outputs = llm.generate(
                    inputs,
                    sampling_params,
                ) 
            aggregate_outputs.append(outputs)

        if not aggregate_outputs:
            breakpoint()
        else:
            print(aggregate_outputs[0])


        # Print the outputs.
        responses = []
        for outputs in aggregate_outputs:
            for output in outputs:
                prompt = output.prompt
                print(f"Prompt: {prompt!r}")
                for i in range(len(output.outputs)):
                    generated_text = output.outputs[i].text
                    responses.append(generated_text)

        all_responses.append({"uid": problem.id, "prompt":inputs , "responses": responses, "base_model": BASE_MODEL, "lora_dir": LORA_DIR})

        with open(saving_file, "w") as f:
            f.write("\n".join(json.dumps(p) for p in all_responses))
        problem.clear()

    print(f"Saving to {saving_file}")

    time.sleep(15)

def execute_evaluate():
    prompt_creator_obj = prompt_creator()
    answer_file = saving_file
    with open(answer_file) as f:
        problem_answers = [json.loads(line) for line in f]

    os.makedirs("results", exist_ok=True)
    saving_file_answer = 'results/induction_results.json'
    # get just the filename
    saving_file_answer = pathlib.Path(saving_file_answer).name 
    saving_file_answer = pathlib.Path("results") / saving_file_answer
    print(f"Saving to {saving_file_answer}")

    problem_that_works_on_training_examples = dict()
    for problem_idx, p in enumerate(tqdm(problem_answers)):
        uid = p["uid"]
        responses = p["responses"]
        print(f"Problem: {uid}")

        codes = []
        for i, response in enumerate(responses):
            parsed_codes = parse_code(response)
            if parsed_codes:
                code = parsed_codes[0]
            else:
                code = ""
            codes.append(code)
        # get the codes

        arc_problem = get_arc_problem(uid)
        # get the problem


        train_verdicts = []

        ##################################
        # The eval mode, Program filtering
        ##################################

        results, output_grids = multi_validate(arc_problem, codes)
        promising_code = []
        promising_code_num = 0
        for idx, result in enumerate(results):
                assert len(result) == len(arc_problem.train_pairs + arc_problem.test_pairs)
                train_verdict = all([verdict for verdict, _ in result[:len(arc_problem.train_pairs)]])
                train_verdicts.append(train_verdict)
                max_ratio = max([ratio for _, ratio in result])
                min_ratio = min([ratio for _, ratio in result])
                icon = "[+]" if train_verdict else "[ ]"
                print(f"    {icon} Code {idx}: on training examples: {train_verdict}, max_ratio: {max_ratio}, min_ratio: {min_ratio}")
                if train_verdict:
                    output_grid = output_grids[idx]

                    # convert to nested lists
                    def numpy_to_nested_list(arr):
                        """Convert numpy array to nested Python list"""
                        if isinstance(arr, np.ndarray):
                            return [numpy_to_nested_list(x) for x in arr]
                        if isinstance(arr, np.integer):
                            return int(arr)
                        return arr
                    
                    output_grid = numpy_to_nested_list(output_grid)
                    problem_that_works_on_training_examples[uid] = [{'attempt_1': output_grid, 'attempt_2': output_grid}]
                    break
                if min_ratio > debug_threshold and promising_code_num<=3:
                    promising_code_num += 1
                    promising_code.append(codes[idx])
                

        # if any(train_verdicts):
        #         continue
        # elif promising_code != []:
        #     for code in promising_code:
        #             debugged_codes = debug_program(code = code, problem = arc_problem,prompt_creator= prompt_creator_obj,llm= llm,tokenizer= tokenizer)
        #             print(f"    debugging code {idx}")
        #             debuged_results, debuged_output_grids = multi_validate(arc_problem, debugged_codes)
        #             debuged_train_verdicts = []
        #             for idx_debug, result in enumerate(debuged_results):
        #                 assert len(debuged_results) == len(arc_problem.train_pairs + arc_problem.test_pairs)
        #                 debuged_train_verdict = all([verdict for verdict, _ in result[:len(arc_problem.train_pairs)]])
        #                 debuged_train_verdicts.append(train_verdict)
        #                 max_ratio = max([ratio for _, ratio in result])
        #                 min_ratio = min([ratio for _, ratio in result])
        #                 icon = "[+]" if train_verdict else "[ ]"
        #                 print(f"    {icon} debugged code {idx}.{idx_debug}: on training examples: {train_verdict}, max_ratio: {max_ratio}, min_ratio: {min_ratio}")
        #             if debuged_train_verdict:
        #                 debuged_output_grid = debuged_output_grids[idx_debug]
        #                 problem_that_works_on_training_examples[uid] = [{'attempt_1': debuged_output_grid, 'attempt_2': None}]


    ##################################
    # Reporting and saving
    ##################################

    print(f"Working on training examples: {len(problem_that_works_on_training_examples)}/{len(problem_answers)}")
    
    print(f"Savings to {saving_file_answer}")
    with open(saving_file_answer, "w") as f:
        f.write(json.dumps(problem_that_works_on_training_examples))


# Let's go!
if __name__ == '__main__':
    get_LLM_output()
    execute_evaluate()