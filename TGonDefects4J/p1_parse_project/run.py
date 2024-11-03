# Integrate all focal methods in defects4j-Lang to see the results

import json
import os
import subprocess
import sys
from extract_focal_method import process_project, process_clazz, write_markdown_info, write_full_info, write_corpus_dataset
from collections import OrderedDict
from tqdm import tqdm

class FileHandlingException(Exception):
    def __init__(self, message="File handling exception"):
        self.message = message
        super().__init__(self.message)

def lang():
    clazz_list = []
    methods_list = []

    for i in range(1, 66):

        if i == 2:
            continue
        try:
            subprocess.run(f'defects4j checkout -p Lang -v {i}f -w lang/lang_{i}', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise e

        clazz_number, methods_number = process_project(f"lang/lang_{i}", "save")
        clazz_list.append(clazz_number)
        methods_list.append(methods_number)

    print(clazz_list)
    print(methods_list)

def handle(number, name="chart", project="Chart", path="."):
    clazz_list = []
    methods_list = []
    if not os.path.exists(f"{name}"):
        os.makedirs(f"{name}")
    # if not os.path.exists(f"{name}_save"):
    #     os.makedirs(f"{name}_save")

    for i in tqdm(range(1, number + 1)):

        if i == 6 and name == "cli":
            continue
        if i == 2 and name == "lang":
            continue
        try:
            subprocess.run(f'defects4j checkout -p {project} -v {i}f -w {path}/{name}/{name}_{i}', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise e

        # clazz_number, methods_number = process_project(f"{path}/{name}/{name}_{i}", f"{path}/{name}_save")
        # clazz_list.append(clazz_number)
        # methods_list.append(methods_number)

    # print(clazz_list)
    # print(methods_list)

def extract(project, append_path, base_path='/efs_data/sy/defect4j/p1_parse_project'):
    root_path = os.path.join(base_path, project)
    contents = os.listdir(root_path)
    result_list, finetune_corpus = [], []
    for each_revision in contents:
        # if each_revision == "lang_65":
        #     continue
        print(f"-------------------{each_revision}-----------------")
        project_path = os.path.join(root_path, each_revision)
        current_path = os.getcwd()
        os.chdir(project_path)
        try:
            result = subprocess.run('defects4j export -p classes.modified', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            error_output = result.stderr.decode('utf-8')

            if output:
                clazz_lines = output.split('\n')
                for clazz_line in clazz_lines:
                    print("class:", clazz_line)
                    clazz_name = clazz_line.split('.')[-1]
                    package_path = '/'.join(clazz_line.strip().split('.')[:-1])
                    for root, dirs, files in os.walk(project_path):
                        for file in files:
                            if package_path in root and clazz_name + ".java" == file:
                                clazz_path = os.path.join(root, file)
                                print(f"class_path: {clazz_path}")
                                result_list_temp, finetune_corpus_temp = process_clazz(clazz_path, each_revision)
                    # if len(result_list_temp) == 0:
                    #     raise FileHandlingException("class can not find.")
                    result_list.extend(result_list_temp)
                    finetune_corpus.extend(finetune_corpus_temp)

            else:
                raise FileHandlingException(f"defects4j can not find class in {project_path}")

        except subprocess.CalledProcessError as e:
            raise e
        finally:
            os.chdir(current_path)

    # result_list = deduplicate(result_list)
    # finetune_corpus = deduplicate(finetune_corpus)
    li = []

    for class_info in result_list:
        for method_info in class_info["methods"]:
            li.append((class_info["package_info"], class_info["class_name"], method_info["signature"],
                       method_info["start"], method_info["end"]))
    li_map = [
        {
            "package": item[0],
            "class_name": item[1],
            "method_name": item[2],
            "start": item[3],
            "end": item[4],
        }
        for item in li
    ]
    project_name = project_path.split('/')[-1]
    write_markdown_info(li_map, append_path + "/" + project + "_info.md")
    write_full_info(result_list, append_path + "/" + project + "_full.json")
    write_corpus_dataset(finetune_corpus, append_path + "/" + project + "_corpus.jsonl")
    print("Done successfully!")
    return len(result_list), len(finetune_corpus)

def deduplicate(datas):
    data_dict = OrderedDict()
    for data in datas:
        key = {"class_name": data['class_name'],
                "package_info": data['package_info'],
                "import_list": data['import_list'],
                "method_info": data['method_info'],
                "source": data['source']}
        data_dict[json.dumps(key, ensure_ascii=False)] = data
    ret_datas = []
    for data in data_dict.values():
        ret_datas.append(data)
    return ret_datas

if __name__ == "__main__":
    # Two Part:
    
    # chart()
    # 1. Initialize the runtime environment
    path = "/efs_data/sy/defect4j/bac1"
    handle(65, "lang", "Lang", path)
    handle(26, "chart", "Chart", path)
    handle(40, "cli", "Cli", path)
    handle(16, "csv", "Csv", path)
    handle(18, "gson", "Gson", path)

    # 2. Extract project information
    # project_names = ["csv", "cli", "gson", "lang", "chart"]
    # project_names = ['lang']
    # class_number = []
    # method_number = []
    # for project_name in project_names:
    #     a, b = extract(project_name, "/efs_data/sy/defect4j/p1_parse_project/save_renew", "/efs_data/sy/defect4j/bac1")
    #     class_number.append(a)
    #     method_number.append(b)
    # print(f"class number: {class_number}")
    # print(f"method number: {method_number}")

