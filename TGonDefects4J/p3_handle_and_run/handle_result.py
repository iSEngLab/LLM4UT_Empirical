import json

def read_json(path):
    with open(path, 'r') as f:
        data = f.read()
    return data

def read_jsonl(path):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def write_jsonl(path, data):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
def handle_predict(predict):
    if isinstance(predict, str):
        return predict
    if isinstance(predict, list):
        predict[0] = "@"
        predict = predict[:-1]
        return ' '.join(predict)

def merge_file(corpus_path, generate_path, append_path):
    corpus = read_jsonl(corpus_path)
    generate = read_jsonl(generate_path)
    # 创建一个字典，以id为键，predict为值
    id_to_predict = {str(item['id']) + item['revision']: handle_predict(item['predict']) for item in generate}

    # 将predict字段内容按照id一一对应的方式加入到a2中的target中
    for item in corpus:
        item['target'] = id_to_predict.get(str(item['id']) + item['revision'], None)
    sorted_corpus = sorted(corpus, key=lambda x: x['revision'])

    write_jsonl(append_path, sorted_corpus)

def merge_file_2(corpus_path, generate_path, append_path):
    corpus = read_jsonl(corpus_path)
    generate = read_jsonl(generate_path)
    for item in corpus:
        for each in generate:
            if item['target'] == each['label']:
                item['target'] = each['predict']
    
    sorted_corpus = sorted(corpus, key=lambda x: x['revision'])

    write_jsonl(append_path, sorted_corpus)
    
if __name__ == "__main__":
    model_name = "qwen2_7b_extract_2"
    corpus_path = "/efs_data/sy/defect4j/p1_parse_project/renew_merged.jsonl"
    generate_path = f"/efs_data/sy/defect4j/p2_pred_test_case/save_result_close/{model_name}_pred.jsonl"
    append_path = f"/efs_data/sy/defect4j/p3_handle_and_run/close_result/{model_name}_result.jsonl"
    merge_file(corpus_path, generate_path, append_path)
    # merge_file_2(corpus_path, generate_path, append_path)
