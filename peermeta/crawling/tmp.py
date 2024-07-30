import json
import random

import jsonlines


samples = []
with jsonlines.open("../data/iclr_2023.jsonl") as reader:
    for line in reader:
        samples.append(line)

with open("iclr_2023_tmp.json", "w") as f:
    json.dump(random.sample(samples, 50), f, indent=4)