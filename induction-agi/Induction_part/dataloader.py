## This file is used to load the data from the files and return the data in the form of a list of problems
import os
import json
import numpy as np
from problem_class import problem


################################################
# The function to load the data
################################################


def load_data(folder_path, mode):
    problems = []
    for filename in os.listdir(folder_path): 
        with open(os.path.join(folder_path, filename), 'r') as file:
            data = json.load(file)
            id = filename.split('.')[0]
            problem_instance = problem(id = id, train = data['train'], test = data['test'], type = mode)
            problems.append(problem_instance)
    return problems