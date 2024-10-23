import jsonlines
from openai import OpenAI
import time
import json
import random
from tqdm import tqdm


def comparing(source_documents, generation_a, generation_b, dataset_name):
    prompt_format = open(f"prompt_comparing_{dataset_name}.txt").read()
    source_text = "\n".join(source_documents)
    prompt_content = prompt_format.replace("{{source_documents}}", source_text).replace("{{generation_a}}",
                                                                                        generation_a).replace(
        "{{generation_b}}", generation_b)
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


if __name__ == "__main__":
    random.seed(42)
    # pair-wise comparison on test samples in the three domains
    model_name = "gpt4"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    with open("../info.json") as f:
        info = json.load(f)

    dataset_names = ["space"]
    for dataset_name in dataset_names:
        print(dataset_name)
        generation_files = info[dataset_name]
        generations_model = {}
        for generation_file in generation_files:
            with open(generation_file["generation_file"]) as f:
                generations_model[generation_file["model_name"]] = json.load(f)

        # constructing pairs
        with open("/home/miao4/punim0521/ModularMetaReview/modular_llama3/hotels/space_generation_result_llama31_70b.json") as f:
            all_samples = json.load(f)

        if dataset_name == "peermeta": # get the shared samples in results from all models
            all_samples_test = []
            for sample in all_samples:
                reference = sample["meta_review"]
                for result in generations_model["modular_llama3"]:
                    if reference == result["meta_review"]:
                        all_samples_test.append(sample)
                        break
            all_samples = all_samples_test

            for model_name, samples in generations_model.items():
                samples_new = []
                for sample_test in all_samples_test:
                    for sample_model in samples:
                        if sample_model["meta_review"] == sample_test["meta_review"]:
                            samples_new.append(sample_model)
                            break
                generations_model[model_name] = samples_new

        for model, results in generations_model.items():
            print(model, len(all_samples), len(results))
            reference_key = ""
            candidate_key = ""
            for generation_file in generation_files:
                if model == generation_file["model_name"]:
                    reference_key = generation_file["reference_key"]
                    candidate_key = generation_file["candidate_key"]
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
        for generation_file in generation_files:
            if generation_file["model_name"] == "modular_llama3":
                reference_key = generation_file["reference_key"]

        for j, result in enumerate(generations_model["modular_llama3"]):
            assert result[reference_key] == all_samples[j][reference_key]
            generations = all_samples[j].get("generations", [])
            if isinstance(result[reference_key], str):
                generations.append({"model": "human", "generation": result[reference_key]})
            if isinstance(result[reference_key], list):
                generations.append({"model": "human", "generation": result[reference_key][0]})
            all_samples[j]["generations"] = generations

        all_samples = random.sample(all_samples, 25)
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
                            {"a": generation_i["model"], "b": generation_j["model"], "better": prediction})
            all_samples[sample_index]["comparisons"] = comparisons
            print(len(generations), len(comparisons))

        with open(f"{dataset_name}_llm_judged.json", "w") as f:
            json.dump(all_samples, f, indent=4)
