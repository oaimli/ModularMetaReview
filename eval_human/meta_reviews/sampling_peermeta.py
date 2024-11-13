import json
import random

random.seed(42)

with open("../../peermeta/data/peermeta_all.json") as f:
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

# split data into two categorizes
samples_diff = []
samples_sim = []
for sample in samples_test:
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

samples_combined = []
for sample in samples_diff + samples_sim:
    instance = {}
    instance["paper_id"] = sample["paper_id"]
    instance["meta_review"] = sample["meta_review"]
    source_documents = []
    source_documents.append({"review_id": "abstract", "content": sample["paper_abstract"], "reply_to": sample["paper_id"]})
    for review in sample["reviews"]:
        source_documents.append({"review_id": review["review_id"], "content": review["comment"], "reply_to": review["reply_to"]})
    # source texts with conversational structures
    instance["source_documents"] = source_documents
    instance["label"] = "test"
    samples_combined.append(instance)

print("samples_combined", len(samples_combined))


with open("info.json") as f:
    all_info = json.load(f)

dataset_name = "peermeta"
generation_infos = all_info[dataset_name]
print(dataset_name)
samples = {}
for generation_info in generation_infos:
    print(generation_info)
    candidate_key = generation_info["candidate_key"]
    reference_key = generation_info["reference_key"]
    with open(generation_info["generation_file"]) as f:
        generations = json.load(f)
    for i, generation in enumerate(generations):
        sample_new = samples.get(f"index_{i}", {})

        # only source texts
        source_documents = []
        for source in sample_new.get("source_documents", []):
            source_documents.append(source["content"])

        for sample_test in samples_combined:
            tmp = []
            for source in sample_test["source_documents"]:
                tmp.append(source["content"])
            if sample_test["meta_review"] == generation[reference_key]:
                paper_id = sample_test["paper_id"]
                sample_new["paper_id"] = paper_id
                # source documents with conversational structures
                sample_new["source_documents"] = sample_test["source_documents"]
        assert sample_new["paper_id"] == ""
        assert sample_new["source_documents"] == []

        tmp = sample_new.get("generations", {})
        tmp["human_reference"] = generation[reference_key]
        tmp[generation_info["model_name"]] = generation[candidate_key]
        sample_new["generations"] = tmp
        samples[f"index_{i}"] = sample_new

indexes = []
for sample_key, sample_value in samples.items():
    source_documents = []
    for source in sample_value["source_documents"]:
        source_documents.append(source["content"])
    source_text_length = len("\n".join(source_documents).split())
    if len(source_documents) <= 10 and source_text_length < 3000:
        indexes.append(sample_key)
print(len(indexes))
indexes_sampled = random.sample(indexes, 20)

samples_sampled = {}
for sample_index in indexes_sampled:
    samples_sampled[sample_index] = samples[sample_index]

with open(f"generations_{dataset_name}.json", "w") as f:
    json.dump(samples_sampled, f, indent=4)
