# Import the package

from prompt_creator import prompt_creator
from vllm import SamplingParams
from execute import parse_code

def debug_program(code, problem, prompt_creator, llm, tokenizer, temperature = 0.8, batch_size = 1, num_of_samples_per_problem = 4):
    # the prompt
    prompt = prompt_creator.create_debugging_prompt(code, problem)
    messages = prompt_creator.prompt_to_message_debug(prompt)

    # turn to valid llm input
    inputs = tokenizer.apply_chat_template([
            {"role":"system", "content":messages[0]["content"]},
            {"role":"user", "content":messages[1]["content"]}
        ], tokenize=False, add_generation_prompt=True)
    input_tokens = tokenizer.apply_chat_template([
            {"role":"system", "content":messages[0]["content"]},
            {"role":"user", "content":messages[1]["content"]}
        ], tokenize=True, add_generation_prompt=True)
    
    # decide the number of debug samples
    

    # Generate

    sampling_params = SamplingParams(temperature=temperature, max_tokens=1536,
                                        n=batch_size)
    aggregate_outputs = []
    for i in range(num_of_samples_per_problem // batch_size):
            
        outputs = llm.generate(
            inputs,
            sampling_params,
        ) 
        aggregate_outputs.append(outputs)
    # Turn the output to code

    debuged_codes = []

    for llm_outputs in aggregate_outputs:  
        for llm_output in llm_outputs:
            for i in range(len(llm_output.outputs)):
                generated_text = llm_output.outputs[i].text
                parsed_codes = parse_code(generated_text)
                if parsed_codes:
                    code = parsed_codes[0]
                else:
                    code = ""
                debuged_codes.append(code)

    return debuged_codes
