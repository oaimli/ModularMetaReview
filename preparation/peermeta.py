import json
import random
import jsonlines

with open("../peermeta/data/peermeta_all.json") as f:
    samples = json.load(f)

samples_test = []
samples_dev = []
samples_train = []
for sample in samples:
    if sample["label"] == "test":
        samples_test.append(sample)
    if sample["label"] == "val":
        samples_dev.append(sample)
    if sample["label"] == "train":
        samples_train.append(sample)
print(len(samples_train), len(samples_dev), len(samples_test))


def split_data(samples_tmp):
    samples_diff = []
    samples_sim = []
    for sample in samples_tmp:
        reviews = sample["reviews"]
        with_disagreements = False
        for i, review_i in enumerate(reviews):
            for j, review_j in enumerate(reviews):
                if j > i:
                    if review_i["rating"] > 0 and review_j["rating"] > 0:
                        dis = review_i["rating"] - review_j["rating"]
                        if abs(dis) >= 5:
                            with_disagreements = True
        if with_disagreements:
            samples_diff.append(sample)
        else:
            samples_sim.append(sample)

    return samples_diff, samples_sim


random.seed(42)

samples_diff, samples_sim = split_data(samples_dev)
samples_diff = random.sample(samples_diff, 25)
samples_sim = random.sample(samples_sim, 25)
samples_combined = []
for sample in samples_diff + samples_sim:
    instance = {}
    instance["meta_review"] = sample["meta_review"]
    source_documents = []
    source_documents.append(sample["paper_abstract"])
    for review in sample["reviews"]:
        source_documents.append(review["comment"])
    instance["source_documents"] = source_documents
    instance["label"] = "valid"
    samples_combined.append(instance)
with jsonlines.open("../datasets/peermeta_dev.jsonl", "w") as writer:
    writer.write_all(samples_combined)

samples_diff, samples_sim = split_data(samples_test)
samples_diff = random.sample(samples_diff, 50)
samples_sim = random.sample(samples_sim, 50)
samples_combined = []
for sample in samples_diff + samples_sim:
    instance = {}
    instance["meta_review"] = sample["meta_review"]
    source_documents = []
    source_documents.append(sample["paper_abstract"])
    for review in sample["reviews"]:
        source_documents.append(review["comment"])
    instance["source_documents"] = source_documents
    instance["label"] = "test"
    samples_combined.append(instance)
with jsonlines.open("../datasets/peermeta_test.jsonl", "w") as writer:
    writer.write_all(samples_combined)

samples_combined = []
for sample in samples_train:
    instance = {}
    instance["meta_review"] = sample["meta_review"]
    source_documents = []
    source_documents.append(sample["paper_abstract"])
    for review in sample["reviews"]:
        source_documents.append(review["comment"])
    instance["source_documents"] = source_documents
    instance["label"] = "train"
    samples_combined.append(instance)
with jsonlines.open("../datasets/peermeta_train.jsonl", "w") as writer:
    writer.write_all(samples_combined)
