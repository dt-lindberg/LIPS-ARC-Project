import numpy as np
import signal
import json

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

################################################
# We create a class to represent a data token
################################################


class problem:
    def __init__(self, id = None, train = None, test = None, type = None, prompt = None, program = None, induction_solution = None,):
        self.id = id
        self.train = train
        self.test = test
        self.type = type
        self.prompt = prompt
        self.program = program
    
    def give_prompt(self, prompt):
        self.prompt = prompt

    def give_program_description(self, text_outputs):
        self.description_program_pair = []
        for text_output in text_outputs:
            print(text_output)
            if text_output['generated_text'].startswith(self.prompt):
                text = text_output[len(self.prompt):].strip()
                output= text
            print(output)
            text_output = output.split("</reasoning>")
            cleaned_program = text_output[1].strip('```').strip('```')
            description_program_pair = {'description': text_output[0], 'program': cleaned_program}
            self.description_program_pair.append(description_program_pair)
    
    
    def clear(self):
        self.program = None
        self.description_program_pair = None
        self.prompt = None
        self.type = None
        # Clear everything except id, test and solution
    
    