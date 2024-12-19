import json
import random
import time
import jsonlines
import numpy as np
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

samples_all = []
for sample_test in samples_test:
    meta_reviews = sample_test["gold_summaries_general"]
    sources = sample_test["source_documents"]

    sample_new = {}
    sample_new["source_documents"] = random.sample(sources, 20)
    # human-written reference
    sample_new["human_references"] = meta_reviews
    sample_new["generation_decomposed"] = ""
    sample_new["steps_decomposed"] = ""
    sample_new["generation_modular"] = ""
    sample_new["steps_modular"] = ""
    samples_all.append(sample_new)

sampled_indexes = []
for sample_index, sample in enumerate(samples_all):
    if sample_index not in indexes_meta_reviews:
        sampled_indexes.append(sample_index)
sampled_indexes = random.sample(sampled_indexes, 10)

# get the samples from the test dataset
samples_sampled = {}
for sample_index, sample in enumerate(samples_all):
    if sample_index in sampled_indexes:
        samples_sampled[f"index_{sample_index}"] = sample
print("Sampled samples", len(samples_sampled))

# statistics of these sampled samples
source_lengths = []
for sample_key, sample_value in samples_sampled.items():
    source_lengths.append(len(" ".join(sample_value["source_documents"]).split()))
print("Average source length", np.mean(source_lengths))

def get_generations_with_modular(samples_sampled):
    return samples_sampled

def get_generations_with_decomposed(samples_sampled):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    # reproduce the intermediate output of decomposed prompting
    for sample_sampled_key in samples_sampled.keys():
        sample_sampled = samples_sampled[sample_sampled_key]
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
        samples_sampled[sample_sampled_key] = sample_sampled

        return samples_sampled


# samples_sampled = get_generations_with_modular(samples_sampled)
# samples_sampled = get_generations_with_decomposed(samples_sampled)
with open("sampled_hotel.json", "w") as f:
    json.dump(samples_sampled, f, indent=4)
