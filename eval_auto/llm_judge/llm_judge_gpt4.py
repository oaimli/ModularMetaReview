import jsonlines
from openai import OpenAI
import time
import json
import random


def comparing(source_documents, generation_a, generation_b, dataset_name):
    prompt_format = open(f"prompt_comparing_{dataset_name}.txt").read()
    source_text = "\n".join(source_documents)
    prompt_content = prompt_format.replace("{{source_documents}}", source_text)
    prompt_content = prompt_content.replace("{{generation_a}}", generation_a)
    prompt_content = prompt_content.replace("{{generation_b}}", generation_b)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system",
                     "content": "Always answer with only the answer without any other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=10
                )
            output = []
            for choice in output_dict.choices:
                tmp = choice.message.content.lower()
                if "a" in tmp:
                    output.append("a")
                if "b" in tmp:
                    output.append("b")
            if len(output) > 7:
                prediction = max(output, key=output.count)
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)

    return prediction


def scoring(samples):
    winning_rates = {}
    elo_scores = {}
    return winning_rates, elo_scores


if __name__ == "__main__":
    random.seed(42)
    # pair-wise comparison on test samples in the three domains
    model_name = "gpt4"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    with open("../info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)
        generation_files = info[dataset_name]
        generations_model = {}
        for generation_file in generation_files:
            with open(generation_file["generation_file"]) as f:
                generations_model[generation_file["model_name"]] = json.load(f)

        # constructing pairs
        all_samples = []
        with jsonlines.open(f"../../datasets/{dataset_name}_test.jsonl") as reader:
            for line in reader:
                all_samples.append(line)

        for model, results in generations_model.items():
            reference_key = ""
            for generation_file in generation_files:
                if model == generation_file["model_name"]:
                    reference_key = generation_file["reference_key"]

            for i, result in enumerate(results):
                assert result[reference_key] == all_samples[i][reference_key]
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
                        prediction = comparing(source_documents, generation_i["generation"], generation_j["generation"], dataset_name)
                        comparisons.append(
                            {"a": generation_i["model"], "b": generation_j["model"], "better": prediction})
            all_samples[sample_index]["comparisons"] = comparisons
        with open(f"{dataset_name}_llm_judged.json", "w") as f:
            json.dump(all_samples, f, indent=4)

        winning_rates, elo_scores = scoring(all_samples)
        print(winning_rates, elo_scores)
