import os
import json

def analyze_results_in_folder(file_path):
    # Initialize result_summary and separate result_counts for each project
    results = {
        'chart': {"CORRECT": 0, "PASS": 0, "TE": 0, "CE": 0, "SE": 0, "total": 0},
        'cli': {"CORRECT": 0, "PASS": 0, "TE": 0, "CE": 0, "SE": 0, "total": 0},
        'csv': {"CORRECT": 0, "PASS": 0, "TE": 0, "CE": 0, "SE": 0, "total": 0},
        'gson': {"CORRECT": 0, "PASS": 0, "TE": 0, "CE": 0, "SE": 0, "total": 0},
        'lang': {"CORRECT": 0, "PASS": 0, "TE": 0, "CE": 0, "SE": 0, "total": 0},
        'total': {"CORRECT": 0, "PASS": 0, "TE": 0, "CE": 0, "SE": 0, "total": 0},
    }

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    # Open the file and read the results
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            result = data.get("result", "")
            # Safely get project name from the revision field
            revision = data.get("revision", "")
            project = revision.split('_')[0] if '_' in revision else ""

            if project in results:
                results[project]["total"] += 1
                results['total']["total"] += 1

                if result == "CORRECT":
                    results[project]["CORRECT"] += 1
                    results[project]["PASS"] += 1
                    results['total']["CORRECT"] += 1
                    results['total']["PASS"] += 1
                elif result == "PASS":
                    results[project]["PASS"] += 1
                    results['total']["PASS"] += 1
                elif result == "TE":
                    results[project]["TE"] += 1
                    results['total']["TE"] += 1
                elif result == "CE":
                    results[project]["CE"] += 1
                    results['total']["CE"] += 1
                elif result == "SE":
                    results[project]["SE"] += 1
                    results['total']["SE"] += 1

    # Generate Markdown table
    markdown_table = "| Model Name | CORRECT Count | PASS Count | TE Count | CE Count | SE Count | Total | CORRECT % | PASS % | TE % | CE % | SE % |\n"
    markdown_table += "|------------|---------------|------------|----------|----------|----------|-------|-----------|--------|------|------|------|\n"

    # Calculate percentages and populate table
    for model_name, result in results.items():
        total = result["total"]
        correct_percentage = (result["CORRECT"] / total) * 100 if total > 0 else 0
        pass_percentage = (result["PASS"] / total) * 100 if total > 0 else 0
        te_percentage = (result["TE"] / total) * 100 if total > 0 else 0
        ce_percentage = (result["CE"] / total) * 100 if total > 0 else 0
        se_percentage = (result["SE"] / total) * 100 if total > 0 else 0

        markdown_table += f"| {model_name} | {result['CORRECT']} | {result['PASS']} | {result['TE']} | {result['CE']} | {result['SE']} | {total} | {correct_percentage:.2f}% | {pass_percentage:.2f}% | {te_percentage:.2f}% | {ce_percentage:.2f}% | {se_percentage:.2f}% |\n"

    # Print the markdown table
    print(markdown_table)
