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
print(indexes_meta_reviews)

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
    sample_new["source_documents"] = random.sample(sources, 10)
    # human-written reference
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
print(sampled_indexes)

# get the samples from the test dataset
samples_sampled = {}
for sample_index, sample in enumerate(samples_all):
    if sample_index in sampled_indexes:
        samples_sampled[f"index_{sample_index}"] = sample
print("Sampled samples", len(samples_sampled))

# statistics of these sampled samples
source_lengths = []
for sample_key, sample in samples_sampled.items():
    source_documents = sample["source_documents"]
    source_lengths.append(len(" ".join(source_documents).split()))
print("Average source length", np.mean(source_lengths))

def get_generations_with_modular(samples_sampled):
    return samples_sampled

# get generated meta-reviews and intermediate steps with decomposed prompting, same as in the llama3_pr/hotels
def get_generations_with_decomposed(samples_sampled):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    for sample_key, sample in samples_sampled.items():
        print("Processing", sample_key)
        source_documents = sample["source_documents"]
        source_text = "\n".join(source_documents)
        prompt_content = f"Please give me sequential steps to write a summary specific for the following reviews on a hotel.\n\n Reviews on a hotel:\n {source_text}\n\nThe steps to write a summary in different lines:"
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to write the steps. Please output the final answer with only the steps in different lines, no other useless content."},
                        {"role": "user",
                         "content": prompt_content}
                        ],
                    n=1
                    )
                actions = output_dict.choices[0].message.content
                break
            except Exception as e:
                if "limit" in str(e):
                    time.sleep(2)

        decomposed_steps = []
        actions_list = actions.split("\n")
        output = ""
        for j, action in enumerate(actions_list):
            step = {"action": action, "output": ""}
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
            decomposed_steps.append(step)

        sample["generation_decomposed"] = output # the output of the last step
        sample["steps_decomposed"] = decomposed_steps
        samples_sampled[sample_key] = sample

    return samples_sampled


# samples_sampled = get_generations_with_modular(samples_sampled)
samples_sampled = get_generations_with_decomposed(samples_sampled)
with open("sampled_hotel.json", "w") as f:
    json.dump(samples_sampled, f, indent=4)
