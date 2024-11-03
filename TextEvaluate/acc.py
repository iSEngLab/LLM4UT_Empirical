import sys


def calc_accuracy(gold_file: str, output_file: str):
    total = 0
    correct = 0

    with open(output_file, 'r') as output_file, open(gold_file, 'r') as gold_file:
        for output_line, gold_line in zip(output_file.readlines(), gold_file.readlines()):
            output = output_line.split("\t")[1]
            gold = gold_line.split("\t")[1]

            total += 1
            if output == gold:
                correct += 1

    print(f"Accuracy: {round(correct * 1.0 / total * 100, 3)}")


if __name__ == '__main__':
    gold_file = sys.argv[1]
    output_file = sys.argv[2]

    calc_accuracy(gold_file, output_file)
