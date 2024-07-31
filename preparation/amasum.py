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
        # selecting samples in the shoes categories
        if "shoes" in " ".join(tmp).lower() and "shoes" in sample["product_meta"]["title"].lower():
            sample["label"] = "test"
            samples.append(sample)
            categories.update(tmp)
            # print(sample["product_meta"]["title"])
            # print(tmp)

print(len(samples))
# print(categories)
# 50

with open("../datasets/amasum_shoes_test.json", "w") as f:
    json.dump(samples, f, indent=4)


test_folder = "../datasets/amasum/min_10_max_100_revs_filt_complete/valid"

samples = []
categories = set([])
for s in os.listdir(test_folder):
    # print(s)
    with open(os.path.join(test_folder, s)) as f:
        sample = json.load(f)
        tmp = sample["product_meta"].get("categories", [])
        # selecting samples in the shoes categories
        if "shoes" in " ".join(tmp).lower() and "shoes" in sample["product_meta"]["title"].lower():
            sample["label"] = "valid"
            samples.append(sample)
            categories.update(tmp)
            # print(sample["product_meta"]["title"])
            # print(tmp)

print(len(samples))
# print(categories)
# 56

with open("../datasets/amasum_shoes_valid.json", "w") as f:
    json.dump(samples, f, indent=4)


test_folder = "../datasets/amasum/min_10_max_100_revs_filt_complete/train"

samples = []
categories = set([])
for s in os.listdir(test_folder):
    # print(s)
    with open(os.path.join(test_folder, s)) as f:
        sample = json.load(f)
        tmp = sample["product_meta"].get("categories", [])
        # selecting samples in the shoes categories
        if "shoes" in " ".join(tmp).lower() and "shoes" in sample["product_meta"]["title"].lower():
            sample["label"] = "train"
            samples.append(sample)
            categories.update(tmp)
            # print(sample["product_meta"]["title"])
            # print(tmp)

print(len(samples))
print(categories)


with open("../datasets/amasum_shoes_train.json", "w") as f:
    json.dump(samples, f, indent=4)
# 428