import json
import random

import jsonlines


samples = []
with jsonlines.open("../data/nips_2022.jsonl") as reader:
    for line in reader:
        samples.append(line)

with open("nips_2022_tmp.json", "w") as f:
    json.dump(random.sample(samples, 50), f, indent=4)