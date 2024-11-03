from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import os
from tqdm import tqdm
import argparse

device = "cuda"

def init_model(model_path, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_one(prompt, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def extract_code_from_source(source):
    # Regular expression to match content between ```java and ```
    match = re.search(r'```java\s+(.*?)\s+```', source, re.DOTALL)
    
    if match:
        # Return the matched code content
        return match.group(1).strip()
    else:
        # If no match is found, return "nothing"
        return "nothing"

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Process some files.')

    # Add the arguments
    parser.add_argument('--file_path', type=str, help='The path to the file under test')
    parser.add_argument('--output_path', type=str, help='The path to output')
    parser.add_argument('--model_path', type=str, help='The path of the model')
    parser.add_argument('--model_name', type=str, help='The name of the model')
    parser.add_argument('--dtype', type=str, default="float32", help='The name of the model')
    parser.add_argument('--data_url', type=str, default="")
    parser.add_argument('--model_url', type=str, default="")
    parser.add_argument('--output_url', type=str, default="")

    # Execute the parse_args() method
    args = parser.parse_args()
    file_path = args.file_path
    output_path = args.output_path
    model_path = args.model_path
    model_name = args.model_name

    # file_path = "/efs_data/sy/LLM4UT-new/LLM4UT-master/datasets/tu/filter_test.json"
    # output_path = "/efs_data/sy/UTBench/p2_pred_test_case/save_result_close"
    # model_path = '/data/Models/qwen/qwen2-7b-instruct'
    # model_name = "llama"

    # model_path = '/data/Models/LLama31/llama-3-8b-intruct'
    # model_path = '/data/Models/LLama31/llama-3.1-8b-instruct'

    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16


    model, tokenizer = init_model(model_path, dtype)
    data = read_jsonl(file_path)

    for index in tqdm(range(0, len(data))):
        source = data[index]['source']

        prompt = f"Given the following Java test case, complete the <AssertPlaceHolder> with an appropriate assertion.\n"
        prompt += f"Your task is to generate the correct assertion statement that should replace the <AssertPlaceHolder>.\n"
        prompt += f"The output should only include the generated assertion statement, wrapped in a Java code cell.\n"
        prompt += f"### Test Case\n{source}"
        
        result = generate_one(prompt, model, tokenizer)

        target = extract_code_from_source(result)
        append_data = {"predict": target, "label":data[index]['target']}

        with open(os.path.join(output_path, model_name+"_pred.jsonl"), "a", encoding="utf-8") as output_file:
            json.dump(append_data, output_file)
            output_file.write('\n')
    
    print("Done!")
