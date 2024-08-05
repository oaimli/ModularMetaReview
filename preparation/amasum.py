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

# convert to the unified format
samples_unified = []
for sample in samples:
    instance = {}
    website_summary = sample["website_summaries"][0]
    meta_review = []
    meta_review.append(website_summary["verdict"])
    meta_review.extend(website_summary["pros"])
    meta_review.extend(website_summary["cons"])
    source_documents = []
    for customer_review in sample["customer_reviews"]:
        source_documents.append(customer_review["text"])
    instance["meta_review"] = " ".join(meta_review)
    instance["source_documents"] = source_documents
    instance["label"] = "test"
    samples_unified.append(instance)

with open("../datasets/amasum_shoes_test.json", "w") as f:
    json.dump(samples_unified, f, indent=4)


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

# convert to the unified format
samples_unified = []
for sample in samples:
    instance = {}
    website_summary = sample["website_summaries"][0]
    meta_review = []
    meta_review.append(website_summary["verdict"])
    meta_review.extend(website_summary["pros"])
    meta_review.extend(website_summary["cons"])
    source_documents = []
    for customer_review in sample["customer_reviews"]:
        source_documents.append(customer_review["text"])
    instance["meta_review"] = " ".join(meta_review)
    instance["source_documents"] = source_documents
    instance["label"] = "valid"
    samples_unified.append(instance)

with open("../datasets/amasum_shoes_valid.json", "w") as f:
    json.dump(samples_unified, f, indent=4)


test_folder = "../datasets/amasum/min_10_max_100_revs_filt_complete/train"

samples = []
categories = set([])
for s in os.listdir(test_folder):
    # print(s)
    with open(os.path.join(test_folder, s)) as f:
        sample = json.load(f)
        tmp = sample["product_meta"].get("categories", [])
        # selecting samples in the shoes categories
        # if "shoes" in " ".join(tmp).lower() and "shoes" in sample["product_meta"]["title"].lower():
        #     sample["label"] = "train"
        #     samples.append(sample)
        #     categories.update(tmp)
        #     # print(sample["product_meta"]["title"])
        #     # print(tmp)
        sample["label"] = "train"
        samples.append(sample)
        categories.update(tmp)

print(len(samples))
print(categories)

# convert to the unified format
samples_unified = []
for sample in samples:
    instance = {}
    website_summary = sample["website_summaries"][0]
    meta_review = []
    meta_review.append(website_summary["verdict"])
    meta_review.extend(website_summary["pros"])
    meta_review.extend(website_summary["cons"])
    source_documents = []
    for customer_review in sample["customer_reviews"]:
        source_documents.append(customer_review["text"])
    instance["meta_review"] = " ".join(meta_review)
    instance["source_documents"] = source_documents
    instance["label"] = "train"
    samples_unified.append(instance)


with open("../datasets/amasum_shoes_train.json", "w") as f:
    json.dump(samples_unified, f, indent=4)
# shoese: 428, all: 25203