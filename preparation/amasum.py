import jsonlines
import json
import os

test_shoes_ids = []
# ids for test samples are from https://github.com/tomhosking/hercules/
with jsonlines.open("../../tmp/hercules-main/data/data_amasum/amasum-eval-shoes/test.jsonl") as reader:
    for line in reader:
        test_shoes_ids.append(line["entity_id"])

test_folder = "../datasets/amasum/min_10_revs_filt_complete/test"
samples = []
for id in test_shoes_ids:
    with open(os.path.join(test_folder, id + ".json")) as f:
        sample = json.load(f)
        sample["entity_id"] = id
        samples.append(sample)
print(len(samples)) # 50

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
    instance["entity_id"] = sample["entity_id"]
    instance["meta_review"] = " ".join(meta_review)
    instance["source_documents"] = source_documents
    instance["label"] = "test"
    samples_unified.append(instance)

with jsonlines.open("../datasets/amasum_shoes_test.jsonl", "w") as writer:
    writer.write_all(samples_unified)



valid_shoes_ids = []
# ids for dev samples are from https://github.com/tomhosking/hercules/
with jsonlines.open("../../tmp/hercules-main/data/data_amasum/amasum-eval-shoes/dev.jsonl") as reader:
    for line in reader:
        valid_shoes_ids.append(line["entity_id"])

valid_folder = "../datasets/amasum/min_10_revs_filt_complete/valid"
samples = []
for id in valid_shoes_ids:
    with open(os.path.join(valid_folder, id + ".json")) as f:
        sample = json.load(f)
        sample["entity_id"] = id
        samples.append(sample)
print(len(samples)) # 50

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
    instance["entity_id"] = sample["entity_id"]
    instance["meta_review"] = " ".join(meta_review)
    instance["source_documents"] = source_documents
    instance["label"] = "valid"
    samples_unified.append(instance)

with jsonlines.open("../datasets/amasum_shoes_valid.jsonl", "w") as writer:
    writer.write_all(samples_unified)


train_folder = "../datasets/amasum/min_10_revs_filt_complete/train"
samples = []
for s in os.listdir(train_folder):
    with open(os.path.join(train_folder, s)) as f:
        sample = json.load(f)
        sample["entity_id"] = s[:-5]
        sample["label"] = "train"
        samples.append(sample)
print(len(samples))

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
    instance["entity_id"] = sample["entity_id"]
    instance["meta_review"] = " ".join(meta_review)
    instance["source_documents"] = source_documents
    instance["label"] = "train"
    samples_unified.append(instance)

with jsonlines.open("../datasets/amasum_shoes_train.jsonl", "w") as writer:
    writer.write_all(samples_unified)
# all: 25203
