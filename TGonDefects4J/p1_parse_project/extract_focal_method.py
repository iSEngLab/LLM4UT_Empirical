import json
import os
from code_parser import *
import javalang
from classes import Markdown
from transformers import AutoTokenizer

def get_type_string(type):
    if type is None:
        return 'void'
    # 处理基本类型和引用类型
    type_str = ''
    if isinstance(type, javalang.tree.ReferenceType):
        type_str = type.name
        if type.arguments:
            args = ', '.join(get_type_string(arg.type) for arg in type.arguments if arg.type is not None)
            type_str += f"<{args}>"
    elif isinstance(type, javalang.tree.BasicType):
        type_str = type.name
    else:
        type_str = str(type)

    # 处理数组类型
    dimensions = ''.join('[]' for _ in range(len(type.dimensions))) if hasattr(type, 'dimensions') else ''
    return f"{type_str}{dimensions}"

def remove_comments(input_string):
    import re
    pattern = r'/\*\*?\s.*?\*/'
    return re.sub(pattern, '', input_string, flags=re.DOTALL)

def process_string(input_str):
    # 将'\n'和'\t'替换为' '
    input_str = input_str.replace('\n', ' ').replace('\t', ' ')
    # 处理 '{' 和 '}'，确保其左右都是空格
    input_str = input_str.replace('{', ' { ').replace('}', ' } ')
    # 将多余的空格替换为一个空格
    input_str = ' '.join(input_str.split())
    return input_str

def filter_by_token_length(source, tokenizer):
    src = source
    src_encode = tokenizer.encode(src)
    if len(src_encode) >= 512:
        return False
    return True

total_number = 0
def get_class_info(file: str):
    global total_number
    file = remove_comments(file)

    # magic replace
    magic_table = {
        "\'{\'": "6416864816840",
        "\'}\'": "6416864816841",
    }
    for k, v in magic_table.items():
        file = file.replace(k, v)

    file_lines = file.splitlines()
    try:
        tree = javalang.parse.parse(file)
        # assert len(tree.types) == 1  # 一个文件只有一个类
    except Exception as e:
        print(e)
        return None, None, None, None
    clazz: javalang.tree.ClassDeclaration = tree.types[0]
    print("processing", clazz.name, clazz.modifiers)
    if not isinstance(clazz, javalang.tree.ClassDeclaration):# or 'abstract' in clazz.modifiers:
        return None, None, None, None

    clazz_name = clazz.name
    class_body = clazz.body

    import_info = [x.path for x in tree.imports]
    package_info = tree.package.name

    construct_list = []
    for item in class_body:
        if type(item) == javalang.tree.ConstructorDeclaration:
            construct_list.append(parse_node(item))

    properties_list = []
    for item in class_body:
        if type(item) == javalang.tree.FieldDeclaration:
            properties_list.append(parse_node(item))

    clazz_info = {
        "clazz_name": clazz_name,
        "package_info": package_info,
        "import_list": import_info,
        "construct_list": construct_list,
        "properties_list": properties_list
    }

    methods: List[MethodDeclaration] = clazz.methods

    method_infos = []
    all_method_infos = [] # 看上去应该是没用的，但还是先不删了

    for method in methods:
        valid_focal_method = 'public' in method.modifiers

        start = method.position[0]

        def get_method_end(method):
            def f(node):
                if hasattr(node, 'children'):
                    for child in node.children:
                        yield from f(child)

                if isinstance(node, list):
                    for child in node:
                        yield from f(child)

                if hasattr(node, 'position'):
                    if node.position is not None:
                        yield node.position[0]

            stmt_set = set(f(method))
            return max(stmt_set) + 1

        while 1:
            if start - 2 >= 0 and file_lines[start - 2].strip().startswith("@"):
                start -= 1
            else:
                break

        end = get_method_end(method)
        content = file_lines[start - 1:end]
        # 缩进处理
        indent = len(content[0]) - len(content[0].lstrip())
        content = [line[indent:] for line in content]
        content = "\n".join(content)

        # 获取方法的返回类型，包括泛型信息和数组
        return_type = get_type_string(method.return_type)

        # 获取方法的参数类型，包括泛型信息和数组
        params = []
        for parameter in method.parameters:
            param_type = get_type_string(parameter.type)
            if parameter.varargs:
                param_type += "..."
            params.append(param_type)

        # 构建方法签名
        method_signature = parse_node(method)
        # method_signature = f"{method.name}({', '.join(params)}) : {return_type}"
        # params, invokes = parse_method_declaration(method)
        info = {
            "signature": method_signature,
            "name": method.name,
            "start": start,
            "end": end,
            "content": content,
            # "params": params,
            # "invokes": invokes
        }
        if valid_focal_method:
            print(f"valid method: {method_signature} ({start}-{end})")
            method_infos.append(info)
        else:
            print(f"invalid method: {method.modifiers} {method_signature} ({start}-{end})")
        all_method_infos.append(info)

    # 还原magic_string
    for k, v in magic_table.items():
        file = file.replace(v, k)

    return clazz_name, file, method_infos, clazz_info


def process_project(project_path, append_path):

    # 准备一个列表来存储结果
    index = 0
    result_list = []
    finetune_corpus = []
    tokenizer = AutoTokenizer.from_pretrained("/data/Models/CodeBert-base")

    for root, dirs, files in os.walk(project_path):
        for file in files:
            if 'package-info.java' in file:
                continue
            if ("src/main/java" in root or "src/java" in root or "source/org/jfree" in root) and file.endswith(".java"):
                print("trying process", os.path.join(root, file))
                with open(os.path.join(root, file), 'r', encoding='windows-1254') as f:
                    data = f.read()

                clazz_name, file_data, method_infos, clazz_info = get_class_info(data)
                if clazz_name is not None:
                    result_list.append({
                        "class_name": clazz_name,
                        "package_info": clazz_info['package_info'],
                        "import_list": clazz_info['import_list'],
                        "construct_list": clazz_info['construct_list'],
                        "properties_list": clazz_info['properties_list'],
                        "content": file_data,
                        "methods": sorted(method_infos, key=lambda x: x["start"]),
                        "lines": len(file_data.splitlines()),
                    })
                    for method in method_infos:
                        source = clazz_name + ' { ' + method['content']
                        for other_method in method_infos:
                            if method != other_method:
                                source += " " + other_method['signature']
                        for propertise in clazz_info['properties_list']:
                            source += " " + propertise
                        source += "}"
                        source = process_string(source)
                        # if filter_by_token_length(source, tokenizer):
                        finetune_corpus.append({
                            "id": index,
                            "class_name": clazz_name,
                            "package_info": clazz_info['package_info'],
                            "import_list": clazz_info['import_list'],
                            "method_info": method,
                            "source": source,
                            "target": ""
                        })
                        index += 1
                        
    print(f"clazz num: {len(result_list)}")
    print(f"method num: {len(finetune_corpus)}")

    # 验证数据集正确性
    def has_overlap(intervals):
        intervals.sort(key=lambda x: x[0])  # 按照起点进行排序
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:  # 如果下一个区间的起点小于上一个区间的终点，说明有重合
                print("overlap:" ,intervals[i], intervals[i - 1])
                return True
        return False

    li = []

    for class_info in result_list:
        for method_info in class_info["methods"]:
            li.append((class_info["package_info"],class_info["class_name"], method_info["signature"],
                       method_info["start"], method_info["end"]))
        if has_overlap([(method_info["start"], method_info["end"]) for method_info in class_info["methods"]]):
            print(f"overlap found! {class_info['class_name']}")
            assert False

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
    write_markdown_info(li_map, append_path + "/" + project_name + "_info.md")
    write_full_info(result_list, append_path + "/" + project_name + "_full.json")
    write_corpus_dataset(finetune_corpus, append_path + "/" + project_name + "_corpus.jsonl")
    print("Done successfully!")
    return len(result_list), len(finetune_corpus)

def process_clazz(clazz_path, revision):
    index = 0
    result_list = []
    finetune_corpus = []
    with open(clazz_path, 'r', encoding='windows-1254') as f:
        data = f.read()
    clazz_name, file_data, method_infos, clazz_info = get_class_info(data)
    if clazz_name is not None:
        result_list.append({
            "class_name": clazz_name,
            "revision": revision,
            "package_info": clazz_info['package_info'],
            "import_list": clazz_info['import_list'],
            "construct_list": clazz_info['construct_list'],
            "properties_list": clazz_info['properties_list'],
            "content": file_data,
            "methods": sorted(method_infos, key=lambda x: x["start"]),
            "lines": len(file_data.splitlines()),
        })
        for method in method_infos:
            source = clazz_name + ' { ' + method['content']
            for other_method in method_infos:
                if method != other_method:
                    source += " " + other_method['signature']
            for propertise in clazz_info['properties_list']:
                source += " " + propertise
            source += "}"
            source = process_string(source)
            # if filter_by_token_length(source, tokenizer):
            finetune_corpus.append({
                "id": index,
                "revision": revision,
                "class_name": clazz_name,
                "package_info": clazz_info['package_info'],
                "import_list": clazz_info['import_list'],
                "method_info": method,
                "source": source,
                "target": ""
            })
            index += 1
            
    print(f"method num: {len(finetune_corpus)}")
    return result_list, finetune_corpus

def write_markdown_info(li_map, file_path):
    with open(file_path, "w") as f:
        markdown = Markdown().table(li_map).build()
        f.write(markdown)

def write_full_info(full_info, file_path):
    data_to_store = {"data": sorted(full_info, key=lambda x: x["class_name"])}
    with open(file_path, "w") as json_file:
        json.dump(data_to_store, json_file)

def write_corpus_dataset(corpus, file_path):
    with open(file_path, 'w') as f:
        for each in corpus:
            json.dump(each, f)
            f.write('\n')

if __name__ == "__main__":
    project_path_1 = "/efs_data/sy/defect4j/p1_parse_project/chart/chart_2"
    # project_path_2 = "/efs_data/sy/defect4j/java_project/jfreechart-1.5.3"
    # project_path_3 = "/efs_data/sy/defect4j/java_project/commons-cli-commons-cli-1.5.0"
    # project_path_4 = "/efs_data/sy/defect4j/java_project/commons-csv-rel-commons-csv-1.9.0"
    # project_path_5 = "/efs_data/sy/defect4j/java_project/gson-gson-parent-2.8.6/gson"
    # append_path = "/efs_data/sy/defect4j/append"
    append_path = "save_new"
    if not os.path.exists(append_path):
        os.makedirs(append_path)
    process_project(project_path_1, append_path)
    # process_project(project_path_2, append_path)
    # process_project(project_path_3, append_path)
    # process_project(project_path_4, append_path)
    # process_project(project_path_5, append_path)
