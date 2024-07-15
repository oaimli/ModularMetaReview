import json
import os

test_folder = "../datasets/amasum/min_10_max_100_revs_filt_complete/test"

samples = []
categories = set([])
for s in os.listdir(test_folder):
    # print(s)
    with open(os.path.join(test_folder, s)) as f:
        sample = json.load(f)
        tmp = sample["product_meta"].get("categories", [])
        if "shoes" in " ".join(tmp).lower() and "shoes" in sample["product_meta"]["title"].lower():
            sample["label"] = "test"
            samples.append(sample)
            categories.update(tmp)
            # print(sample["product_meta"]["title"])
            # print(tmp)

print(len(samples))
# print(categories)

with open("../datasets/amasum_shoes_test.json", "w") as f:
    json.dump(samples, f, indent=4)
