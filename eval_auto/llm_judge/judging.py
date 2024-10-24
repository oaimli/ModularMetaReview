import jsonlines
from openai import OpenAI
import time
import json
import random


def comparing(source_documents, generation_a, generation_b, dataset_name):
    prompt_format = open(f"prompt_comparing_{dataset_name}.txt").read()
    source_text = "\n".join(source_documents)
    prompt_content = prompt_format.replace("{{source_documents}}", source_text).replace("{{generation_a}}",
                                                                                        generation_a).replace(
        "{{generation_b}}", generation_b)
    print(len(source_text.split()), len(generation_a.split()), len(generation_b.split()), len(prompt_content.split()))
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system",
                     "content": "Always output only the answer without any other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=10
                )
            output = []
            for choice in output_dict.choices:
                tmp = choice.message.content.lower()
                print(tmp)
                if "a" in tmp:
                    output.append("a")
                if "b" in tmp:
                    output.append("b")
            if len(output) > 5:
                prediction = max(output, key=output.count)
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(all_candidates)
    return prediction


if __name__ == "__main__":
    random.seed(42)
    # pair-wise comparison on test samples in the three domains
    model_name = "gpt4"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    with open("../all.json") as f:
        info = json.load(f)

    dataset_names = ["space", "peermeta", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)
        generation_infos = info[dataset_name]
        generations_model = {}
        for generation_info in generation_infos:
            with open(generation_info["generation_file"]) as f:
                generations_model[generation_info["model_name"]] = json.load(f)

        # constructing pairs
        all_samples = []
        with jsonlines.open(f"../../datasets/{dataset_name}_test.jsonl") as reader:
            for line in reader:
                all_samples.append(line)

        for model, results in generations_model.items():
            print(model, len(all_samples), len(results))
            reference_key = ""
            candidate_key = ""
            for generation_info in generation_infos:
                if model == generation_info["model_name"]:
                    reference_key = generation_info["reference_key"]
                    candidate_key = generation_info["candidate_key"]
                    break
            assert reference_key != "" and candidate_key != ""

            for i, result in enumerate(results):
                # print(result[reference_key])
                # print(all_samples[i][reference_key])
                assert result[reference_key] == all_samples[i][reference_key]
                generations = all_samples[i].get("generations", [])
                generations.append({"model": model, "generation": result[candidate_key]})
                all_samples[i]["generations"] = generations

        # add human reference into comparison
        reference_key = ""
        for generation_info in generation_infos:
            if generation_info["model_name"] == "llama3_pr_naive":
                reference_key = generation_info["reference_key"]

        for j, result in enumerate(generations_model["llama3_pr_naive"]):
            assert result[reference_key] == all_samples[j][reference_key]
            generations = all_samples[j].get("generations", [])
            if isinstance(result[reference_key], str):
                generations.append({"model": "human", "generation": result[reference_key]})
            if isinstance(result[reference_key], list):
                generations.append({"model": "human", "generation": result[reference_key][0]})
            all_samples[j]["generations"] = generations

        # construct comparison pairs
        all_samples = random.sample(all_samples, 10)
        for sample_index, sample in enumerate(all_samples):
            generations = sample["generations"]
            source_documents = sample["source_documents"]
            comparisons = []
            for i in range(len(generations)):
                for j in range(len(generations)):
                    if j > i:
                        generation_i = generations[i]
                        generation_j = generations[j]
                        prediction = comparing(source_documents, generation_i["generation"], generation_j["generation"],
                                               dataset_name)
                        comparisons.append(
                            {"model_a": generation_i["model"], "generation_a": generation_i["generation"],
                             "model_b": generation_j["model"], "generation_b": generation_j["generation"],
                             "better": prediction, "source_documents": source_documents})
            all_samples[sample_index]["comparisons"] = comparisons
            print(sample_index, len(generations), len(comparisons))

        with open(f"{dataset_name}_llm_judged.json", "w") as f:
            json.dump(all_samples, f, indent=4)
