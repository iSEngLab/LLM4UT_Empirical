import os
import subprocess


def extract(project, append_path, base_path='/efs_data/sy/defect4j/p1_parse_project'):
    root_path = os.path.join(base_path, project)
    contents = os.listdir(root_path)
    result_list, finetune_corpus = [], []
    for each_revision in contents:
        # if each_revision != "cli_9":
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
