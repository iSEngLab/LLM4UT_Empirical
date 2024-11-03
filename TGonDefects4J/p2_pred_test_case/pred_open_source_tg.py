from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import os
from tqdm import tqdm
import argparse

device = "cuda"
temperature = 1
top_k = 0
top_p = 1
do_sample = True
num_return_sequences = 1

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

    # Create attention_mask
    attention_mask = model_inputs.input_ids.ne(tokenizer.pad_token_id).to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id  # Use eos_token_id as padding token
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
    parser.add_argument('--complete_path', type=str, default='null')

    # Execute the parse_args() method
    args = parser.parse_args()
    file_path = args.file_path
    output_path = args.output_path
    model_path = args.model_path
    model_name = args.model_name
    complete_path = args.complete_path

    data = read_jsonl(file_path)
    if complete_path == "null":
        pass
    else:
        reference = read_jsonl(complete_path)
        new_data = []
        for each in reference:
            if each['predict'] == "nothing":
                for item in data:
                    if item['id'] == each['id'] and item['revision'] == each['revision']:
                        new_data.append(item)
                        break
        print("filter nothing files: ")
        print(len(new_data))
        data = new_data
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    model, tokenizer = init_model(model_path, dtype)

    for index in tqdm(range(0, len(data))):
        source = data[index]['source']
        signature = data[index]['method_info']['signature']
        prompt = f"Please write JUnit 3 unit test for function: `{signature}`, output completed test code in a java code cell.\n The full content is given below: \n {source}"
        
        result = generate_one(prompt, model, tokenizer)

        target = extract_code_from_source(result)
        append_data = {"id": data[index]['id'],
                "revision": data[index]['revision'],
                "predict": target,
                "source": result}

        with open(os.path.join(output_path, model_name+"_pred.jsonl"), "a", encoding="utf-8") as output_file:
            json.dump(append_data, output_file)
            output_file.write('\n')
    
    print("Done!")
