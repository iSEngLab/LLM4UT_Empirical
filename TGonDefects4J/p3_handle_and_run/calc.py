import os
import json

def analyze_results_in_folder(folder_path):
    result_summary = {}

    # Get all jsonl files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_result.jsonl"):
            model_name = filename.replace("_result.jsonl", "")
            file_path = os.path.join(folder_path, filename)

            # Initialize counters for this model
            result_counts = {
                "CORRECT": 0,
                "PASS": 0,
                "TE": 0,
                "CE": 0,
                "SE": 0,
            }
            total_count = 0

            # Process each line in the JSONL file
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    result = data.get("result", "")
                    total_count += 1

                    if result == "CORRECT":
                        result_counts["CORRECT"] += 1
                        result_counts["PASS"] += 1
                    elif result == "PASS":
                        result_counts["PASS"] += 1
                    elif result == "TE":
                        result_counts["TE"] += 1
                    elif result == "CE":
                        result_counts["CE"] += 1
                    elif result == "SE":
                        result_counts["SE"] += 1

            # Store the results for this model
            result_summary[model_name] = {
                "counts": result_counts,
                "total": total_count,
            }

    # Generate Markdown table
    markdown_table = "| Model Name | CORRECT Count | PASS Count | TE Count | CE Count | SE Count | Total | CORRECT % | PASS % | TE % | CE % | SE % |\n"
    markdown_table += "|------------|---------------|------------|----------|----------|----------|-------|-----------|--------|------|------|------|\n"

    for model_name, results in result_summary.items():
        counts = results["counts"]
        total = results["total"]
        correct_percentage = (counts["CORRECT"] / total) * 100 if total > 0 else 0
        pass_percentage = (counts["PASS"] / total) * 100 if total > 0 else 0
        te_percentage = (counts["TE"] / total) * 100 if total > 0 else 0
        ce_percentage = (counts["CE"] / total) * 100 if total > 0 else 0
        se_percentage = (counts["SE"] / total) * 100 if total > 0 else 0

        markdown_table += f"| {model_name} | {counts['CORRECT']} | {counts['PASS']} | {counts['TE']} | {counts['CE']} | {counts['SE']} | {total} | {correct_percentage:.2f}% | {pass_percentage:.2f}% | {te_percentage:.2f}% | {ce_percentage:.2f}% | {se_percentage:.2f}% |\n"

    # Print the markdown table
    print(markdown_table)

# Example usage
analyze_results_in_folder("temp_result")
