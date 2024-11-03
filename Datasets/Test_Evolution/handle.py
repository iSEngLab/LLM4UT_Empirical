import json
import random

# Set random seed to ensure reproducibility
random.seed(42)

def remove_surrogates(s):
    return ''.join(c if not 0xD800 <= ord(c) < 0xE000 else '' for c in s)

def filter_json(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return

    keys_to_keep = {'focal_src', 'focal_tgt', 'test_src', 'test_tgt'}
    filtered_data = []
    for item in data:
        filtered_item = {key: remove_surrogates(item[key]) for key in keys_to_keep if key in item}
        filtered_data.append(filtered_item)

    try:
        # Convert data to JSON string
        json_string = json.dumps(filtered_data, ensure_ascii=False, indent=4)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(json_string)
    except UnicodeEncodeError as e:
        print(f"Unicode encoding error: {e}")

    print(f"Filtered data written to '{output_path}'.")

# Usage example
filter_json('train.json', 'filter_train.json')
filter_json('test.json', 'filter_test.json')

# Read the original training data
with open('filter_train.json', 'r') as file:
    data = json.load(file)

# Calculate validation set size based on the ratio
valid_size = len(data) // 9  # train:valid:test is 8:1:1, so validation set is 1/9 of the training set

# Shuffle data randomly
random.shuffle(data)

# Split data into new training set and validation set
valid_data = data[:valid_size]
train_data = data[valid_size:]

# Save the new training set and validation set
with open('filter_valid.json', 'w') as file:
    json.dump(valid_data, file, indent=4)

with open('filter_train.json', 'w') as file:
    json.dump(train_data, file, indent=4)

print(f'New training set length: {len(train_data)}, validation set length: {len(valid_data)}')
