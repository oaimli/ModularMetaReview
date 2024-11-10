import json
import random

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

random.seed(42)  # to ensure sampled test samples are the same
samples_diff = random.sample(samples_diff, 50)
samples_sim = random.sample(samples_sim, 50)
samples_combined = []
sampled_paper_ids = []
for sample in samples_diff + samples_sim:
    instance = {}
    instance["paper_id"] = sample["paper_id"]
    instance["meta_review"] = sample["meta_review"]
    source_documents = []
    source_documents.append(sample["paper_abstract"])
    for review in sample["reviews"]:
        source_documents.append(review["comment"])
    instance["source_documents"] = source_documents
    instance["label"] = "test"
    samples_combined.append(instance)
    sampled_paper_ids.append(sample["paper_id"])

with open("info.json") as f:
    all_info = json.load(f)

print(sampled_paper_ids)
# load data for decomposed prompting
generation_info_decomposed = all_info["peermeta"][1]
generation_file_decomposed = generation_info_decomposed["generation_file"]
candidate_key_decomposed = generation_info_decomposed["candidate_key"]
reference_key_decomposed = generation_info_decomposed["reference_key"]
with open(generation_file_decomposed) as f:
    samples_decomposed = json.load(f)

# load data for modular prompting
generation_info_modular = all_info["peermeta"][0]
generation_file_modular = generation_info_modular["generation_file"]
candidate_key_modular = generation_info_modular["candidate_key"]
reference_key_modular = generation_info_modular["reference_key"]
with open(generation_file_modular) as f:
    samples_modular = json.load(f)

samples_all = []
for sample_origin, sample_modular, sample_decomposed in zip(samples_combined, samples_modular, samples_decomposed):
    paper_id = sample_origin["paper_id"]
    meta_review = sample_origin["meta_review"]
    source_documents = sample_origin["source_documents"]

    sources_origin = sample_origin["source_documents"]
    sources_modular = sample_modular["source_documents"]
    sources_decomposed = sample_decomposed["source_documents"]
    print(sources_origin[0])
    print(sources_modular[0])
    print(sources_decomposed[0])
    assert sources_origin[0] == sources_modular[0] == sources_decomposed[0]

    sample_new = {}
    sample_new["paper_id"] = paper_id
    sample_new["source_documents"] = source_documents
    # human-written reference
    sample_new["meta_review"] = meta_review

    # generated meta-review from decomposed prompting
    sample_new["generation_decomposed"] = sample_decomposed[candidate_key_decomposed]
    # steps from decomposed prompting
    decomposed_steps = []
    for action in sample_decomposed["generated_steps"]:
        decomposed_steps.append({"action": action, "output": ""})

    # generated meta-review from modular prompting
    sample_new["generation_modular"] = sample_modular[candidate_key_modular]
    # steps from modular prompting
    modular_steps = sample_modular["categorization_pairs"]
    sample_new["steps_modular"] = modular_steps

    if len(source_documents) <= 10:
        samples_all.append(sample_new)
print(len(samples_all))

samples_sampled = random.sample(samples_all, 9)

# reproduce the intermediate output of decomposed prompting

with open("sampled.json", "w") as f:
    json.dump(samples_sampled, f)
