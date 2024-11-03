import json
import random
import os

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def split_repos(repo_data):
    repos = list(repo_data.keys())
    random.shuffle(repos)
    total = len(repos)
    print(f"total repos: {total}")
    size_train = round(0.8 * total)
    size_validation = round(0.1 * total)
    
    train_repos = repos[:size_train]
    validation_repos = repos[size_train:size_train + size_validation]
    test_repos = repos[size_train + size_validation:]
    
    return train_repos, validation_repos, test_repos

def split_repos_new(repo_data):
    repos = list(repo_data.keys())
    random.shuffle(repos)
    
    # Count the total number of data entries
    total_data_count = sum(len(repo_data[repo]) for repo in repos)
    target_train_size = round(0.8 * total_data_count)
    target_validation_size = round(0.1 * total_data_count)
    target_test_size = total_data_count - target_train_size - target_validation_size
    
    train_repos, validation_repos, test_repos = [], [], []
    train_size, validation_size, test_size = 0, 0, 0
    
    for repo in repos:
        repo_size = len(repo_data[repo])
        
        # Choose which set to put the current repo in to get as close to the target ratio as possible
        if train_size + repo_size <= target_train_size:
            train_repos.append(repo)
            train_size += repo_size
        elif validation_size + repo_size <= target_validation_size:
            validation_repos.append(repo)
            validation_size += repo_size
        else:
            test_repos.append(repo)
            test_size += repo_size
    
    print(f"Train size: {train_size}, Validation size: {validation_size}, Test size: {test_size}")
    return train_repos, validation_repos, test_repos

def process_and_save_data(repo_data, train_repos, validation_repos, test_repos, base_dir):
    for category, repos in zip(['train', 'eval', 'test'], [train_repos, validation_repos, test_repos]):
        category_dir = os.path.join(base_dir, category)
        # os.makedirs(category_dir, exist_ok=True)
        with open(category_dir + '.jsonl', 'w') as f:        
            for repo in repos:
                for file_path in repo_data[repo]:
                    try:
                        data = load_json_file(file_path)
                        # Creating a unique filename to avoid overwriting
                        # basename = os.path.basename(file_path)
                        # save_path = os.path.join(category_dir, basename)
                        # save_data(data, save_path)
                        f.write(json.dumps(data) + '\n')
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

def main(repo_data_file, base_dir):
    print("start")
    repo_data = load_json_file(repo_data_file)
    print("load data over")
    train_repos, validation_repos, test_repos = split_repos_new(repo_data)
    print("split data over")
    process_and_save_data(repo_data, train_repos, validation_repos, test_repos, base_dir)
    print("all work done")

# Example usage
repo_data_file = 'filter/list_dedup.json'  # Path to the JSON file containing repo data
base_dir = 'dedup'  # Base directory to save train/validation/test datasets
main(repo_data_file, base_dir)
