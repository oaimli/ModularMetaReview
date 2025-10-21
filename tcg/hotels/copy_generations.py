import pickle
import os.path
import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def combine_aspects(summaries_aspect: List) -> str:
    source_text = "\n".join(summaries_aspect)
    print(source_text)
    prompt_content = f"Please write a summary for the following opinions.\n\n Opinions on a hotel:\n {source_text}\n\nThe output summary:"
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system",
                     "content": "You are requested to do summarization. Please output the final answer with only the summary, no other useless content."},
                    {"role": "user",
                     "content": prompt_content}
                ],
                n=1
            )
            output = output_dict.choices[0].message.content
            break
        except Exception as e:
            print(e)
            if "limit" in str(e):
                time.sleep(2)
    print(output)
    return output


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # get test data
    test_samples = []
    with jsonlines.open("../../datasets/space_test.jsonl") as reader:
        for line in reader:
            test_samples.append(line)

    # get source reviews, and mapping results
    reviews_folder = "../ZS-Summ-GPT3/saved-data/space/reviews"
    source_index = {}
    for filename in os.listdir(reviews_folder):
        with open(os.path.join(reviews_folder, filename), "r") as f:
            source_text = f.read()
        source_index[source_text] = filename[:-4]

    mappings = []
    for sample in test_samples:
        source_text = "\n".join(sample["source_documents"])
        source_text_words = source_text.split()
        inter_counts = []
        for tmp in source_index.keys():
            tmp_words = tmp.split()
            inter = list(set(source_text_words).intersection(set(tmp_words)))
            inter_counts.append(len(inter))
        if max(inter_counts) > 2000:
            mappings.append(source_index[list(source_index.keys())[inter_counts.index(max(inter_counts))]])
        else:
            mappings.append("none")

    # TCG
    generation_file = "../ZS-Summ-GPT3/saved-data/space/pickle/summaries-pkl/tcg.pkl"
    with open(generation_file, 'rb') as f:
        generations = pickle.load(f)

    results = []
    for sample, mapping in zip(test_samples, mappings):
        if mapping != "none":
            generation = generations[mapping]
            sample["generated_summary_service"] = generation["service"]
            sample["generated_summary_rooms"] = generation["rooms"]
            sample["generated_summary_location"] = generation["location"]
            sample["generated_summary_food"] = generation["food"]
            sample["generated_summary_cleanliness"] = generation["cleanliness"]
            sample["generated_summary_building"] = generation["building"]
            sample["generated_summary_general"] = combine_aspects(
                [generation["service"], generation["rooms"], generation["location"], generation["food"],
                 generation["cleanliness"], generation["building"]])
            results.append(sample)
            # print(sample)
        else:
            sample["generated_summary_service"] = ""
            sample["generated_summary_rooms"] = ""
            sample["generated_summary_location"] = ""
            sample["generated_summary_food"] = ""
            sample["generated_summary_cleanliness"] = ""
            sample["generated_summary_building"] = ""
            sample["generated_summary_general"] = ""
            sample["comment"] = "error"
            results.append(sample)

    print(len(results))
    output_dir = "../../results/greg_gpt3_space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_tcg.json", "w") as f:
        json.dump(results, f, indent=4)

    # First-TCG
    generation_file = "../ZS-Summ-GPT3/saved-data/space/pickle/summaries-pkl/tcg-first.pkl"
    with open(generation_file, 'rb') as f:
        generations = pickle.load(f)
    results = []
    for sample, mapping in zip(test_samples, mappings):
        if mapping != "none":
            generation = generations[mapping]
            sample["generated_summary_service"] = generation["service"]
            sample["generated_summary_rooms"] = generation["rooms"]
            sample["generated_summary_location"] = generation["location"]
            sample["generated_summary_food"] = generation["food"]
            sample["generated_summary_cleanliness"] = generation["cleanliness"]
            sample["generated_summary_building"] = generation["building"]
            sample["generated_summary_general"] = combine_aspects(
                [generation["service"], generation["rooms"], generation["location"], generation["food"],
                 generation["cleanliness"], generation["building"]])
            results.append(sample)
            # print(sample)
        else:
            sample["generated_summary_service"] = ""
            sample["generated_summary_rooms"] = ""
            sample["generated_summary_location"] = ""
            sample["generated_summary_food"] = ""
            sample["generated_summary_cleanliness"] = ""
            sample["generated_summary_building"] = ""
            sample["generated_summary_general"] = ""
            sample["comment"] = "error"
            results.append(sample)

    print(len(results))
    output_dir = "../../results/greg_gpt3_space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_tcg_first.json", "w") as f:
        json.dump(results, f, indent=4)

    # QG
    generation_file = "../ZS-Summ-GPT3/saved-data/space/pickle/summaries-pkl/qg.pkl"
    with open(generation_file, 'rb') as f:
        generations = pickle.load(f)
    results = []
    for sample, mapping in zip(test_samples, mappings):
        if mapping != "none":
            generation = generations[mapping]
            sample["generated_summary_service"] = generation["service"]
            sample["generated_summary_rooms"] = generation["rooms"]
            sample["generated_summary_location"] = generation["location"]
            sample["generated_summary_food"] = generation["food"]
            sample["generated_summary_cleanliness"] = generation["cleanliness"]
            sample["generated_summary_building"] = generation["building"]
            sample["generated_summary_general"] = combine_aspects(
                [generation["service"], generation["rooms"], generation["location"], generation["food"],
                 generation["cleanliness"], generation["building"]])
            results.append(sample)
            # print(sample)
        else:
            sample["generated_summary_service"] = ""
            sample["generated_summary_rooms"] = ""
            sample["generated_summary_location"] = ""
            sample["generated_summary_food"] = ""
            sample["generated_summary_cleanliness"] = ""
            sample["generated_summary_building"] = ""
            sample["generated_summary_general"] = ""
            sample["comment"] = "error"
            results.append(sample)

    print(len(results))
    output_dir = "../../results/greg_gpt3_space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_qg.json", "w") as f:
        json.dump(results, f, indent=4)

    # TQG
    generation_file = "../ZS-Summ-GPT3/saved-data/space/pickle/summaries-pkl/tqg.pkl"
    with open(generation_file, 'rb') as f:
        generations = pickle.load(f)
    results = []
    for sample, mapping in zip(test_samples, mappings):
        if mapping != "none":
            generation = generations[mapping]
            sample["generated_summary_service"] = generation["service"]
            sample["generated_summary_rooms"] = generation["rooms"]
            sample["generated_summary_location"] = generation["location"]
            sample["generated_summary_food"] = generation["food"]
            sample["generated_summary_cleanliness"] = generation["cleanliness"]
            sample["generated_summary_building"] = generation["building"]
            sample["generated_summary_general"] = combine_aspects(
                [generation["service"], generation["rooms"], generation["location"], generation["food"],
                 generation["cleanliness"], generation["building"]])
            results.append(sample)
            # print(sample)
        else:
            sample["generated_summary_service"] = ""
            sample["generated_summary_rooms"] = ""
            sample["generated_summary_location"] = ""
            sample["generated_summary_food"] = ""
            sample["generated_summary_cleanliness"] = ""
            sample["generated_summary_building"] = ""
            sample["generated_summary_general"] = ""
            sample["comment"] = "error"
            results.append(sample)

    print(len(results))
    output_dir = "../../results/greg_gpt3_space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_tqg.json", "w") as f:
        json.dump(results, f, indent=4)
