import json
import random
import time
import jsonlines
from openai import OpenAI


random.seed(42)

# filter out samples that are used in the pair-wise human evaluation on meta-reviews
with open("../meta_reviews/generations_space.json") as f:
    samples_meta_reviews = json.load(f)
indexes_meta_reviews = []
for sample_index in samples_meta_reviews.keys():
    indexes_meta_reviews.append(int(sample_index.split("_")[1]))

# load the original test dataset
samples_test = []
with jsonlines.open("../../datasets/space_test.jsonl") as reader:
    for line in reader:
        samples_test.append(line)

# load the generations of the two approaches
with open("info_hotel.json") as f:
    all_info = json.load(f)

# load data for decomposed prompting
generation_info_decomposed = all_info["space"][1]
generation_file_decomposed = generation_info_decomposed["generation_file"]
candidate_key_decomposed = generation_info_decomposed["candidate_key"]
reference_key_decomposed = generation_info_decomposed["reference_key"]
with open(generation_file_decomposed) as f:
    samples_decomposed = json.load(f)

# load data for modular prompting
generation_info_modular = all_info["space"][0]
generation_file_modular = generation_info_modular["generation_file"]
candidate_key_modular = generation_info_modular["candidate_key"]
reference_key_modular = generation_info_modular["reference_key"]
with open(generation_file_modular) as f:
    samples_modular = json.load(f)

samples_all = []
for sample_modular, sample_decomposed, sample_test in zip(samples_modular, samples_decomposed, samples_test):
    reference_index = random.choice([0, 1, 2])
    meta_review_modular = sample_modular[reference_key_modular][reference_index]
    meta_review_decomposed = sample_decomposed[reference_key_decomposed][reference_index]
    meta_review_test = sample_test["gold_summaries_general"][reference_index]
    sources_modular = sample_modular["source_documents"]
    sources_decomposed = sample_decomposed["source_documents"]
    sources_test = sample_test["source_documents"]
    assert sources_modular[0] == sources_decomposed[0] == sources_test[0] and meta_review_modular == meta_review_decomposed == meta_review_test

    sample_new = {}
    sample_new["source_documents"] = sources_modular
    # human-written reference
    sample_new["human_reference"] = meta_review_modular
    # generated meta-review from decomposed prompting
    sample_new["generation_decomposed"] = sample_decomposed[candidate_key_decomposed]
    # steps from decomposed prompting
    decomposed_steps = []
    for action in sample_decomposed["generated_steps"].split("\n"):
        if action.strip() != "":
            decomposed_steps.append({"action": action, "output": ""})
    sample_new["steps_decomposed"] = decomposed_steps

    # generated meta-review from modular prompting
    sample_new["generation_modular"] = sample_modular[candidate_key_modular]
    # steps from modular prompting
    modular_steps = sample_modular["categorization_pairs"]
    sample_new["steps_modular"] = modular_steps

print("Possible samples", len(samples_all))

sampled_indexes = []
for sample_index, sample in enumerate(samples_all):
    if sample_index not in indexes_meta_reviews:
        sampled_indexes.append(sample_index)
sampled_indexes = random.sample(sampled_indexes, 10)
print(sampled_indexes)

# get the paper ids from the origin test set
samples_sampled = []
for sample_index, sample in samples_all:
    if sample_index in sampled_indexes:
        samples_sampled.append(sample)
print(len(samples_sampled))

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    )

# reproduce the intermediate output of decomposed prompting
for i, sample_sampled in enumerate(samples_sampled):
    decomposed_steps = sample_sampled["steps_decomposed"]
    source_text = "\n".join(sample_sampled["source_documents"])
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



with open("sampled_hotel.json", "w") as f:
    json.dump(samples_sampled, f, indent=4)
