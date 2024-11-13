import json
import random
import time
from openai import OpenAI

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
for sample_modular, sample_decomposed in zip(samples_modular, samples_decomposed):
    meta_review_modular = sample_modular[reference_key_modular]
    meta_review_decomposed = sample_decomposed[reference_key_decomposed]
    sources_modular = sample_modular["source_documents"]
    sources_decomposed = sample_decomposed["source_documents"]
    assert sources_modular[0] == sources_decomposed[0] and meta_review_modular == meta_review_decomposed

    sample_new = {}
    for sample_test in samples_combined:
        tmp = []
        for source in sample_test["source_documents"]:
            tmp.append(source["content"])
        if sample_test["meta_review"] == meta_review_modular and tmp == sources_modular:
            sample_new["paper_id"] = sample_test["paper_id"]
            sample_new["source_documents"] = sample_test["source_documents"]
            break
    assert sample_new["paper_id"] != ""
    assert sample_new["source_documents"] != []

    # human-written reference
    sample_new["human_reference"] = meta_review_modular

    # generated meta-review from decomposed prompting
    sample_new["generation_decomposed"] = sample_decomposed[candidate_key_decomposed]
    # steps from decomposed prompting
    decomposed_steps = []
    for action in sample_decomposed["generated_steps"].split("\n"):
        if action != "":
            decomposed_steps.append({"action": action, "output": ""})
    sample_new["steps_decomposed"] = decomposed_steps

    # generated meta-review from modular prompting
    sample_new["generation_modular"] = sample_modular[candidate_key_modular]
    # steps from modular prompting
    modular_steps = sample_modular["categorization_pairs"]
    sample_new["steps_modular"] = modular_steps

    if len(sources_modular) <= 10:
        samples_all.append(sample_new)

print("Possible samples", len(samples_all))

random.seed(42)
# samples_sampled = random.sample(samples_all, 9)
sampled_paper_ids = []
with open("../meta_reviews/generations_peermeta.json") as f:
    generations_peermeta = json.load(f)
    for key in generations_peermeta.keys()[:9]:
        sample = generations_peermeta[key]
        sampled_paper_ids.append(sample["paper_id"])

# get the paper ids from the origin test set
samples_sampled = []
for paper_id in enumerate(sampled_paper_ids):
    for sample in samples_all:
        if sample["paper_id"] == paper_id:
            samples_sampled.append(sample)
            break

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    )

# reproduce the intermediate output of decomposed prompting
for i, sample_sampled in enumerate(samples_sampled):
    source_documents = []
    for source in sample_sampled["source_documents"]:
        source_documents.append(source["content"])
    decomposed_steps = sample_sampled["steps_decomposed"]
    source_text = "\n".join(source_documents)
    output = ""
    for j, step in enumerate(decomposed_steps):
        action = step["action"]
        if j == 0:
            prompt_content = f"{source_text}\nPlease follow the instruction below and give your output.\n {action}\nThe output:"
        else:
            prompt_content = f"{output}\nPlease follow the instruction below and give your output.\n {action}\nThe output:"
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to follow the instruction and only generate the requested output."},
                        {"role": "user",
                         "content": prompt_content}
                        ],
                    n=1
                    )
                output = output_dict.choices[0].message.content
                break
            except Exception as e:
                if "limit" in str(e):
                    time.sleep(2)
        step["output"] = output
        print(step)
        decomposed_steps[j] = step

    sample_sampled["steps_decomposed"] = decomposed_steps
    samples_sampled[i] = sample_sampled

with open("sampled.json", "w") as f:
    json.dump(samples_sampled, f, indent=4)
