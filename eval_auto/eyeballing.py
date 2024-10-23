import numpy as np
import json


if __name__ == "__main__":
    with open("all.json") as f:
        info = json.load(f)

    dataset_names = ["space", "peermeta", "amasum_shoes"]

    for dataset_name in dataset_names:
        print(dataset_name)
        with open(f"{dataset_name}_summac_source.json") as f:
            all_scores_summac_source = json.load(f)
        generations_info = info[dataset_name]
        outputs_combined = {}
        scores_summac_source_reference = all_scores_summac_source["human_reference"]
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)

            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            scores_summac_source_generation = all_scores_summac_source[generation_file]

            with open(generation_file) as f:
                samples = json.load(f)
            i = 0
            for sample, score_generation, score_reference in zip(samples, scores_summac_source_generation, scores_summac_source_reference):
                tmp = outputs_combined.get(str(i), {})
                if isinstance(sample[reference_key], str):
                    tmp["reference"] = [sample[reference_key], score_reference]
                else:
                    tmp["reference"] = [sample[reference_key][0], score_reference]

                existing_candidates = tmp.get("generations", [])
                existing_candidates.append((generation_info["model_name"], sample[candidate_key], score_generation))
                tmp["generations"] = existing_candidates
                tmp["source_documents"] = sample["source_documents"]
                outputs_combined[str(i)] = tmp
                i += 1

        output_file = f"{dataset_name}_generations_full.json"
        with open(output_file, "w") as f:
            json.dump(outputs_combined, f, indent=4)

    for dataset_name in dataset_names:
        print(dataset_name)
        generations_info = info[dataset_name]
        outputs_combined = {}
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            generation_file = generation_file[:-5] + "_truncated.json"
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            with open(generation_file) as f:
                samples = json.load(f)
            for i, sample in enumerate(samples):
                tmp = outputs_combined.get(str(i), {})
                if isinstance(sample[reference_key], str):
                    tmp["reference"] = sample[reference_key]
                else:
                    tmp["reference"] = sample[reference_key][0]
                existing_candidates = tmp.get("generations", [])
                existing_candidates.append((generation_info["model_name"], sample[candidate_key]))
                tmp["generations"] = existing_candidates
                outputs_combined[str(i)] = tmp

        output_file = f"{dataset_name}_generations_truncated.json"
        with open(output_file, "w") as f:
            json.dump(outputs_combined, f, indent=4)