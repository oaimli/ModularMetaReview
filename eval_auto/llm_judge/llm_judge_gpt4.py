import jsonlines
from openai import OpenAI
import time
import json
import random


def parsing_result(output):
    with open("output_tmp.jsonl", "w") as f:
        f.write(output.strip())
    tmp = []
    try:
        with jsonlines.open("output_tmp.jsonl") as reader:
            for line in reader:
                tmp.append(line)
    except jsonlines.InvalidLineError as err:
        print("Jsonlines parsing error,", err)
    return tmp


def comparing(source_documents, generation_a, generation_b):
    prompt_format = open("prompt_scientific_gpt4.txt").read()
    review_text = "\n".join(metas_generated)
    prompt_content = prompt_format.replace("{{metas_generated}}", review_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system",
                     "content": "Always answer with only the summary in JSON Lines, no other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=5
                )
            output = []
            for choice in output_dict.choices:
                tmp = parsing_result(choice.message.content)
                if len(tmp) > 0:
                    output = tmp
                    break
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(output)
    meta_review = ""
    if len(output) > 0:
        if "meta_review" in output[0].keys():
            meta_review = output[0]["meta_review"]
    print(meta_review)

    return meta_review

def scoring(samples):
    winning_rates = {}
    elo_scores = {}
    return winning_rates, elo_scores



if __name__ == "__main__":
    random.seed(42)
    # pair-wise comparison on test samples in the three domains
    model_name = "gpt4"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    result_folder = "../results/"

    # peermeta, 30
    generations_peermeta = {}
    with open("") as f:
        generations_peermeta["gpt_4o_vanilla"] = json.load(f)
    with open("") as f:
        generations_peermeta["gpt_4o_logical"] = json.load(f)
    # constructing pairs
    all_samples = []
    with jsonlines.open("../../datasets/peermeta_test.jsonl") as reader:
        for line in reader:
            all_samples.append(line)
    for model, results in generations_peermeta.items():
        for i, result in enumerate(results):
            assert result["meta_review"] == all_samples[i]["meta_review"]
            generations = all_samples[i].get("generations", [])
            generations.append({"model": model, "generation": result["generated_meta_review"]})
            all_samples[i]["generations"] = generations
    all_samples = random.sample(all_samples, 30)
    for sample_index, sample in enumerate(all_samples):
        generations = sample["generations"]
        source_documents = sample["source_documents"]
        comparisons = []
        for i in range(len(generations)):
            for j in range(len(generations)):
                if j > i:
                    generation_i = generations[i]
                    generation_j = generations[j]
                    prediction = comparing(source_documents, generation_i["generation"], generation_j["generation"])
                    comparisons.append({"a": generation_i["model"], "b": generation_j["model"], "better": prediction})
        all_samples[sample_index]["comparisons"] = comparisons
    with open("peermeta_llm_judged.json", "w") as f:
        json.dump(all_samples, f)

    winning_rates, elo_scores = scoring(all_samples)
    print("PeerMeta", winning_rates, elo_scores)

    # space, 20
    generations_space = {}
    with open("") as f:
        generations_space["gpt_4o_vanilla"] = json.load(f)
    with open("") as f:
        generations_space["gpt_4o_logical"] = json.load(f)
    # constructing pairs
    all_samples = []
    with jsonlines.open("../../datasets/space_test.jsonl") as reader:
        for line in reader:
            all_samples.append(line)
    for model, results in generations_space.items():
        for i, result in enumerate(results):
            assert result["gold_summaries_general"][0] == all_samples[i]["gold_summaries_general"][0]
            generations = all_samples[i].get("generations", [])
            generations.append({"model": model, "generation": result["generated_summary_general"]})
            all_samples[i]["generations"] = generations
    all_samples = random.sample(all_samples, 20)
    for sample_index, sample in enumerate(all_samples):
        generations = sample["generations"]
        source_documents = sample["source_documents"]
        comparisons = []
        for i in range(len(generations)):
            for j in range(len(generations)):
                if j > i:
                    generation_i = generations[i]
                    generation_j = generations[j]
                    prediction = comparing(source_documents, generation_i["generation"], generation_j["generation"])
                    comparisons.append({"a": generation_i["model"], "b": generation_j["model"], "better": prediction})
        all_samples[sample_index]["comparisons"] = comparisons
    with open("space_llm_judged.json", "w") as f:
        json.dump(all_samples, f)

    winning_rates, elo_scores = scoring(all_samples)
    print("SPACE", winning_rates, elo_scores)

    # amasum-shoes, 20
    generations_amasum = {}
    with open("") as f:
        generations_amasum["gpt_4o_vanilla"] = json.load(f)
    with open("") as f:
        generations_amasum["gpt_4o_logical"] = json.load(f)
    # constructing pairs
    all_samples = []
    with jsonlines.open("../../datasets/amasum_shoes_test.jsonl") as reader:
        for line in reader:
            all_samples.append(line)
    for model, results in generations_amasum.items():
        for i, result in enumerate(results):
            assert result["meta_review"] == all_samples[i]["meta_review"]
            generations = all_samples[i].get("generations", [])
            generations.append({"model": model, "generation": result["generated_meta_review"]})
            all_samples[i]["generations"] = generations
    all_samples = random.sample(all_samples, 20)
    for sample_index, sample in enumerate(all_samples):
        generations = sample["generations"]
        source_documents = sample["source_documents"]
        comparisons = []
        for i in range(len(generations)):
            for j in range(len(generations)):
                if j > i:
                    generation_i = generations[i]
                    generation_j = generations[j]
                    prediction = comparing(source_documents, generation_i["generation"], generation_j["generation"])
                    comparisons.append({"a": generation_i["model"], "b": generation_j["model"], "better": prediction})
        all_samples[sample_index]["comparisons"] = comparisons
    with open("amasum_shoes_llm_judged.json", "w") as f:
        json.dump(all_samples, f)

    winning_rates, elo_scores = scoring(all_samples)
    print("AmaSum-shoes", winning_rates, elo_scores)