import jsonlines
from openai import OpenAI
import time
import json
import random
import numpy as np

def scoring_faithfulness(source_documents, generation, dataset_name):
    prompt_format = open(f"prompt_comparing_{dataset_name}.txt").read()
    source_text = "\n".join(source_documents)
    prompt_content = prompt_format.replace("{{source_documents}}", source_text).replace("{{generation_summary}}",
                                                                                        generation)
    # print(len(source_text.split()), len(generation.split()), len(prompt_content.split()))
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
                # print(tmp)
                try:
                    if 0<= float(tmp.strip()) <=1:
                        output.append(float(tmp))
                except ValueError:
                    continue

            if len(output) > 5:
                prediction = np.mean(output)
                # print(prediction)
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
    client = OpenAI(api_key="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")

    with open("info.json") as f:
        info = json.load(f)

    # dataset_names = ["space", "peermeta", "amasum_shoes"]
    dataset_names = ["space"]
    for dataset_name in dataset_names:
        print(dataset_name)
        generation_infos = info[dataset_name]

        output_scores = {}
        generation_file = generation_infos[0]["generation_file"]
        print("human reference")
        reference_key = generation_infos[0]["reference_key"]
        source_key = "source_documents"
        with open(generation_file) as f:
            samples = json.load(f)
        scores = []
        for sample in samples:
            if "comment" not in sample.keys():
                if isinstance(sample[reference_key], str):
                    reference = sample[reference_key]
                else:
                    reference = sample[reference_key][0]  # SPACE has multiple references
                score = scoring_faithfulness(sample[source_key], reference, dataset_name)
                scores.append(score)
        print("faithfulness", np.mean(scores))
        output_scores["human_reference"] = scores

        for generation_info in generation_infos:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            with open(generation_file) as f:
                all_samples = json.load(f)
            reference_key = generation_info["reference_key"]
            candidate_key = generation_info["candidate_key"]

            scores = []
            for sample in all_samples:
                if "comment" not in sample.keys():
                    candidate = sample[candidate_key]
                    source_documents = sample["source_documents"]
                    score = scoring_faithfulness(source_documents, candidate, dataset_name)
                    scores.append(score)
            print("faithfulness", np.mean(scores))
            output_scores[generation_file] = scores

        with open(f"{dataset_name}_llm_scored_full_tmp.json", "w") as f:
            json.dump(output_scores, f, indent=4)
