import os
import json
from collections import OrderedDict

def merge_jsonl_files(input_dir, output_file):
    data_dict = OrderedDict()

    # Find all files in the input directory that match the pattern {}_corpus.jsonl
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_dir):
            if filename.endswith("_corpus.jsonl"):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        outfile.write(line)

if __name__ == "__main__":
    # Provide directory path and output file path
    # input_directory = f"save_ori"
    # output_file_path = f"ori_corpus.jsonl"

    input_directory = f"save_dedup"
    output_file_path = f"dedup_corpus.jsonl"

    # Call the function
    merge_jsonl_files(input_directory, output_file_path)
