import os
import json
from transformers import AutoTokenizer
import time
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

lower_threshold = 50
upper_threshold = 200
sample_number = 200

def append_json_to_jsonl(json_file_path, jsonl_file_path):
    # Open the JSON file and read the data
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Open the JSONL file and append the data
    with open(jsonl_file_path, 'a') as jsonl_file:
        # Convert JSON data to string and add to the file
        jsonl_file.write(json.dumps(data) + '\n')

def filter_by_token_length(input_dict, tokenizer):
    # Traverse the root directory
    target = input_dict['target']
    src = input_dict['src_fm_fc_ms_ff']
    target_encode = tokenizer.encode(target)
    if len(target_encode) >= 512 or len(target_encode) < 16:
        return False
    src_encode = tokenizer.encode(src)
    if len(src_encode) >= 2048 or len(src_encode) < 64:
        return False
    return True

def filter_by_construct(input_dict):
    # Filter out data that does not meet general rules (only keep data that starts with @Test public void and does not contain structures like throw)
    target = input_dict['target']
    # Check if the string starts with @Test public void
    if not target.startswith("@Test public void test"):
        return False

    # Find the end position of the first ()
    start_index = target.find('(')
    end_index = target.find(')', start_index)

    # If () is not found, or there are no non-space characters after () or it is not {
    if start_index == -1 or end_index == -1 or end_index + 1 >= len(target) or target[end_index + 2] != '{':
        return False

    return True
                        

def loop_dir(data_dir, data_save_path):
    modified_dirs = []  # Record the subdirectories that have been operated on
    total_data_number = 0
    reduce_by_token_length = 0
    reduce_by_construct = 0
    reduce_by_under_threshold = 0
    reduce_by_over_threshold = 0
    rest_number = 0
    reduce_by_dedup = 0
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("/data/Models/CodeBert-base")
    print("start handle ...")
    data_project_dir = {}

    unique_files = {}
    src_fm_fc_ms_ff_dict = defaultdict(list)

    for current_dir, dirs, files in tqdm(os.walk(data_dir)):
        if not dirs:
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(current_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                    
                    src_fm_fc_ms_ff = data.get('src_fm_fc_ms_ff')
                    if src_fm_fc_ms_ff:
                        if src_fm_fc_ms_ff not in unique_files:
                            unique_files[src_fm_fc_ms_ff] = file_path
                        else:
                            reduce_by_dedup += 1
                            # print(f"Duplicate found: {src_fm_fc_ms_ff} in {file_path} and {unique_files[src_fm_fc_ms_ff]}")

    for current_dir, dirs, files in tqdm(os.walk(data_dir)):
        if not dirs:
            total_data_number += len(files)
            valid_files = []

            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(current_dir, file)

                    if file_path in unique_files.values():
                        with open(file_path, 'r', encoding='utf-8') as json_file:
                            data = json.load(json_file)

                        if not filter_by_construct(data):
                            reduce_by_construct += 1
                        elif not filter_by_token_length(data, tokenizer):
                            reduce_by_token_length += 1
                        else:
                            valid_files.append(file_path)

            number_of_file_after_filter = len(valid_files)
            if number_of_file_after_filter < lower_threshold:
                reduce_by_under_threshold += number_of_file_after_filter
                modified_dirs.append({current_dir: f"under {lower_threshold}"})
                continue
            elif number_of_file_after_filter >= upper_threshold:
                files_to_keep = set(random.sample(valid_files, sample_number))
                valid_files = [file for file in valid_files if file in files_to_keep]
                reduce_by_over_threshold += (number_of_file_after_filter - len(files_to_keep))
                modified_dirs.append({current_dir: f"over {upper_threshold}"})
            
            rest_number += len(valid_files)
            data_project_dir[current_dir.split('/')[-1]] = valid_files

    if os.path.exists(data_save_path):
        with open(data_save_path, 'r') as f:
            record = json.load(f)
    else:
        record = {}

    record.update(data_project_dir)
    with open(data_save_path, 'w') as f:
        json.dump(record, f, indent=4)

    end_time = time.time()
    print("done ...")
    print(f"spend time: {end_time - start_time:.2f}s")
    print("-=" * 20 + " statistics " + "-=" * 20)
    print(f"64 < src token length < 2048, 16 < target token length < 512")
    print(f"lower threshold: {lower_threshold}, upper threshold: {upper_threshold}, sample number: {sample_number}")
    print(f"data dir: {data_dir}")
    print(f"total_data_number: {total_data_number}")
    print(f"reduce_by_token_length: {reduce_by_token_length}")
    print(f"reduce_by_construct: {reduce_by_construct}")
    print(f"reduce_by_dedup: {reduce_by_dedup}")
    print(f"reduce_by_under_threshold: {reduce_by_under_threshold}")
    print(f"reduce_by_over_threshold: {reduce_by_over_threshold}")
    print(f"rest_number: {rest_number}")
    print("-=" * 46)
    with open('modified_file.json', 'w') as f:
        json.dump(modified_dirs, f, indent=4)

if __name__ == "__main__":
    dir_under_filters = "/efs_data/sy/LLM4UT-new/LLM4UT-master/datasets/tg/train"
    loop_dir(dir_under_filters, "filter/list_dedup.json")
