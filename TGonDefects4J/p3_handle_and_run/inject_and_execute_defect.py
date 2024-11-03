import json
import os
import subprocess
import shutil
import re
import javalang
import argparse

def read_jsonl(path):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def write_jsonl(path, datas):
    with open(path, 'w') as f:
        for data in datas:
            f.write(json.dumps(data) + '\n')

def process_string(input_str):
    # Check if the string ends with "}"
    if input_str.endswith("}"):
        return input_str
    else:
        # Find the last ";"
        last_semicolon_index = input_str.rfind(";")
        if last_semicolon_index != -1:
            # Remove everything after the last ";" and add "}"
            return input_str[:last_semicolon_index + 1] + "}"
        else:
            # If there is no ";", directly add "}"
            return input_str + "}"

def syntax_verify(java_code):
    # java_code = java_code.replace("@Test", "")
    pattern = r"public void.*"
    match = re.search(pattern, java_code, re.DOTALL)
    if match:
        java_code = match.group(0)
    java_code = process_string(java_code)
    # Surround with a class to make it parsable by javalang
    final_code = f"class mock {{ {java_code}}}"
    # print(final_code)
    try:
        tree = javalang.parse.parse(final_code)
        method_name = tree.types[0].methods[0].name
        # print(tree)
    except Exception as e :
        return f"Syntax Error : {e}", None
    index = java_code.find(method_name)
    if index != -1:
        code = java_code[:index] + "test" + method_name + java_code[index + len(method_name):]
        return code, "test" + method_name
    return code, None

def extract_test_file(test_path):
    with open(test_path, 'r') as f:
        code = f.read()
    tree = javalang.parse.parse(code)
    imports = [x.path for x in tree.imports]
    return imports

def inject_test_class(test_path, code, package, imports, class_name, original_class_name):
    # Construct a JUnit3 structured test class
    test_code = ""
    test_code += f"package {package};\n\n"
    for import_sentence in imports:
        if "org.junit.Assert" not in import_sentence and "org.junit.Test" not in import_sentence:
            test_code += f"import {import_sentence};\n"
    frame_imports = ['junit.framework.TestCase', 'junit.framework.Test', 'junit.framework.TestSuite', 'junit.textui.TestRunner', 'java.io.*', 'java.util.*']
    for import_sentence in frame_imports:
        test_code += f"import {import_sentence};\n"
    test_code += "\n"
    
    test_code += f"import {package}.*;\n\n"

    test_code += f"public class {class_name} extends TestCase {{\n"
    for line in code.split('\n'):
        test_code += "    " + line + '\n'
    test_code += "}\n"
    with open(test_path, 'w') as f:
        f.write(test_code)

def recover_test_class(test_path):
    os.remove(test_path)

def inject_test_method(test_path, code):
    with open(test_path, 'r') as f:
        test_code = f.read()
    try:
        end_position = find_last_brace_position(test_code)
    except:
        return "Env Syntax Error"
    
    modified_code = test_code[:end_position] + code + test_code[end_position:]
    with open(test_path, 'w') as f:
        f.write(modified_code)
    return test_code

def recover_test_method(test_path, code):
    with open(test_path, 'w') as f:
        f.write(code)

def find_last_brace_position(java_code):
    stack = []
    last_brace_position = None

    # Find the position of public class
    public_class_index = java_code.find("public class")
    if public_class_index == -1:
        raise Exception("public class not found")

    # Traverse characters starting from public class
    for i in range(public_class_index, len(java_code)):
        if java_code[i] == '{':
            stack.append(i)
        elif java_code[i] == '}':
            stack.pop()
            if not stack:
                last_brace_position = i
                break

    return last_brace_position

def env_verify():
    # execute mvn clean compile test, make sure correctness of env
    try:
        print("-=-=-=-=-=-=-=-=-= env verify -=-=-=-=-=-=-=-=-=-=")
        subprocess.run('defects4j test', shell=True, check=True)
        print("env ready")
    except subprocess.CalledProcessError as e:
        print("env check fail: ", e)
        raise e

def fix_env():
    now_path = os.getcwd()
    project_name = now_path.split('/')[-2]

def execute_test_file(package, class_name, method_name):
    try:
        print("-=-=-=-=-=-=-=-=-= defects4j test -=-=-=-=-=-=-=-=-=-=")
        print(f'defects4j test -t {package}.{class_name}::{method_name}')
        result = subprocess.run(f'defects4j test -t {package}.{class_name}::{method_name}', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        error_output = result.stderr.decode('utf-8')
        if output.find("Failing tests: 0") == -1:
            return "TE"
        return "AC"
    except subprocess.CalledProcessError as e:
        print("test wrong: ", e)
        return "CE"

def coverage_test_file(package, class_name, method_name):
    try:
        print("-=-=-=-=-=-=-=-=-= defects4j coverage -=-=-=-=-=-=-=-=-=-=")
        result = subprocess.run(f'defects4j coverage -t {package}.{class_name}::{method_name}', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        pattern = r"Lines covered:\s*(\d+)"
        match = re.search(pattern, output)
        if match:
            # Extract the matched number
            number = match.group(1)
            if number != '0':
                return "CORRECT"
            else: 
                return "PASS"
        else:
            return "PASS"
    except subprocess.CalledProcessError as e:
        print("coverage wrong: ", e)
        return "PASS"

def whole_process(root_path, undertest_file_path, backup_path):
    env_clean(root_path)
    env_verify(root_path)
    test_datas = read_jsonl(undertest_file_path)
    result = []
    for test_data in test_datas:
        env_clean(root_path)
        code_file, test_class_name, package_info = construct_test_file(test_data)

        middle_path = os.path.join(root_path, "src/test/java")
        package_path = package_info.replace('.', '/')
        test_path = os.path.join(package_path, test_class_name + ".java")
        inject_path = os.path.join(middle_path, test_path)
        inject_test_file(code_file, inject_path)

        # backup_path = os.path.join(backup_path, test_path)
        # backup_test_file(inject_path, backup_path)

        temp_result = execute_test_file(test_class_name)
        result.append(temp_result)
        parse_test_result()
    print(result)
    print(len([a for a in result if a]))
    print(len(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some files.')

    # Add the arguments
    parser.add_argument('undertest_file_path', type=str, help='The path to the file under test')
    parser.add_argument('model_name', type=str, help='The name of the model')

    # Execute the parse_args() method
    args = parser.parse_args()

    projects_path = "/efs_data/sy/defect4j/p1_parse_project"
    # projects_path = "/efs_data/sy/defect4j/buggy_bac1"
    # undertest_file_path = "/efs_data/sy/defect4j/p3_handle_and_run/close_result/openai_result.jsonl"
    # model_name = "temp"
    undertest_file_path = args.undertest_file_path
    model_name = args.model_name
    append_path = f"/efs_data/sy/defect4j/p3_handle_and_run/temp_result/{model_name}_result.jsonl"

    datas = read_jsonl(undertest_file_path)
    current_directory = os.getcwd()
    # print(project_name)
    for index in range(4736, len(datas)):
        data = datas[index]
        print(f"Index: {index}, Revision: {data['revision']}, Id: {data['id']}")
        project_name = data['revision'].split('_')[0]
        root_path = os.path.join(projects_path, project_name + '/' + data['revision'])
        pre_root_path = os.getcwd()
        os.chdir(root_path)
        # if pre_root_path != root_path:
        #     env_verify()
        code = data['target']
        code, method_name = syntax_verify(code)
        if "Syntax Error" in code:
            data['result'] = "SE"
            # print(f"Result: {data['result']}")
            # continue
        else:
            package_path = data['package_info'].replace('.', '/')
            class_name = data['class_name']
            full_path = ""
            imports = data['import_list']
            for root, dirs, files in os.walk(root_path):
                if root.endswith(package_path):
                    if 'gson' not in root:
                        if 'test' in root and 'test-classes' not in root and 'target' not in root and 'build' not in root:
                            full_path = root
                    else:
                        if 'gson/src/test/' in root and 'target' not in root:
                            full_path = root
            if project_name == "chart":
                inject_class_name = "Unique"+class_name+"Tests"
            else:
                inject_class_name = "Unique"+class_name+"Test"

            inject_path = os.path.join(full_path, inject_class_name + '.java')
            print(inject_path)
            inject_test_class(inject_path, code, data['package_info'], imports, inject_class_name, class_name)

            data['result'] = execute_test_file(data['package_info'], inject_class_name, method_name)
        
            if data['result'] == "AC":
                data['result'] = coverage_test_file(data['package_info'], inject_class_name, method_name)

            recover_test_class(inject_path)

        assert "result" in data.keys()
        
        print(f"Result: {data['result']}")
        with open(append_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    os.chdir(current_directory)
    # write_jsonl(append_path, datas)
    print("Done")