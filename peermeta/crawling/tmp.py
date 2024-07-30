import jsonlines


with jsonlines.open("../data/iclr_2023.jsonl") as reader:
    for line in reader:
        print(line)