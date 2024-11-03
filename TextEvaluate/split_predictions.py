import json
import argparse
import os.path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    contents = []
    with open(args.predictions, "r", encoding="utf-8") as prediction_file:
        for line in prediction_file.readlines():
            line = line.strip()

            contents.append(json.loads(line))

    with open(os.path.join(args.output_path, "test.gold"), "w", encoding="utf-8") as gold_file, open(
            os.path.join(args.output_path, "test.output"), "w", encoding="utf-8") as output_file:

        for idx, content in enumerate(contents):
            gold_file.write(f"{idx}\t{content['label']}\n")
            output_file.write(f"{idx}\t{content['predict']}\n")


if __name__ == "__main__":
    main()
