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
# Import the standard modules we need.

from prompt_creator import prompt_creator
from dataloader import load_data
from problem_class import problem, apply_program
# Import the inductor and prompt_creator from the previous snippets.


# Load the configuration file
with open('Induction_part/config.json', 'r') as config_file:
    config = json.load(config_file)

##################################
# The model directory
##################################

BASE_MODEL = config["BASE_MODEL"]
LORA_DIR = config["LORA_DIR"]

##################################
# Configs
##################################


num_of_samples_per_problem = config["num_of_samples_per_problem"]
# Here the num_of_samples_per_problem is the total number of program samples we want for a simple problem.
TENSOR_PARALLEL = config["TENSOR_PARALLEL"]
mode = config["MODE"]
prompt_engineering = config["prompt_engineering"]

##################################
# Load the tokenizer
##################################


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


#################################################
# Set the device and load the model and create a pipeline
#################################################

device = "cuda" if torch.cuda.is_available() else "cpu"


llm = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
text_generater = pipeline('text-generation', model=llm, tokenizer=tokenizer, device=0)
    

#################################################
# Date directory
#################################################

folderpath = 'ARC-AGI/data/' + mode
    
#################################################
# Induction
#################################################

def give_accuracy(problems):
    correct = 0
    total = 0
    for problem in problems:
        for example in problem.test:
            total += 1
            if len(problem.program) == 0:
                continue
            if example['output'] == problem.induction_solution:
                correct += 1
    return correct/total


def main():
    # Create an inductor object
    inductor_obj = inductor(text_generater, tokenizer, device)
    
    # Create a prompt creator object
    prompt_creator_obj = prompt_creator(device = device, mode = mode)

    # Load the data
    problems = load_data(mode)

    # Create prompt
    for problem in problems:
        prompt = prompt_creator_obj.create_prompt_from_test_task(problem)
        if prompt_engineering:
            prompt = prompt_creator_obj.add_prompt_engineering(prompt)
        problem.give_prompt(prompt)
    
    # Induce the programs
    for problem in tqdm(problems):
        text_output = inductor_obj.model_sample(problem, num_of_samples_per_problem)
        problem.give_program_description(text_output)
        problem.filter_program()
        problem.give_induction_solution(apply_program(problem.test[0]['input'], problem.description_program_pair[0]['program']))
        problem.clear()

    if mode == 'training':
        print('Training Accuracy:', give_accuracy(problems))
    elif mode == 'evaluation':
        pass

    # Save the programs
    for problem in problems:

        with open('Programs' + '/' + mode + '/' + problem.id + '.json', 'w') as f:
            i = 0
            for description_program_pair in problem.description_program_pair:
                i += 1
                f.write(f'the {i}th description program\n')
                json.dump(description_program_pair['description'], f)
                f.write('\n\nThe program:\n')
                json.dump(description_program_pair['program'], f)
                f.write('\n')

if __name__ == '__main__':
    main()
else:
    print('Induction part is not running as the main program.')
