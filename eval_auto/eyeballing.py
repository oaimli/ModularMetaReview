import numpy as np
import json


if __name__ == "__main__":
    with open("all.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)
        generations_info = info[dataset_name]
        outputs_combined = {}
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            with open(generation_file) as f:
                samples = json.load(f)
            for i, sample in enumerate(samples):
                tmp = outputs_combined.get(str(i), {})
                tmp["reference"] = sample[reference_key]
                existing_candidates = tmp.get(candidate_key, [])
                existing_candidates.append({generation_info["model_name"]: sample[candidate_key]})
                tmp["generations"] = existing_candidates
                outputs_combined[str(i)] = tmp

        output_file = f"{dataset_name}_generations.json"
        with open(output_file, "w") as f:
            json.dump(outputs_combined, f, indent=4)