import json
import jsonlines


samples = []
with jsonlines.open("../data/iclr_2023.jsonl") as reader:
    for line in reader:
        print(line)

with open("tmp.json", "w") as f:
    json.dump(samples, f)