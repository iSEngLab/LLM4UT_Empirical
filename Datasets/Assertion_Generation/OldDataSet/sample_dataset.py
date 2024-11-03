import json


def main():
    filename = "assert_train_old.jsonl"
    sample_num = 500

    with open(filename, "r", encoding="utf-8") as f, open(filename.split(".")[0] + f"_{sample_num}" + ".jsonl", "w",
                                                          encoding="utf-8") as output_file:
        for i in range(sample_num):
            line = f.readline()
            content = json.loads(line)
            output_file.write(f"{json.dumps(content)}\n")


if __name__ == "__main__":
    main()
