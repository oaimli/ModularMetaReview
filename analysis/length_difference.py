import json
import numpy as np


def analyzing(targets, sources):
    lengths = []
    for target in targets:
        lengths.append(len(target.split()))
    print(np.mean(lengths), np.max(lengths), np.min(lengths), np.std(lengths))




if __name__ == "__main__":
    with open("../eval_auto/info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)

        facets = []
        if dataset_name == "peermeta":
            facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
        if dataset_name == "space":
            facets = ["Building", "Cleanliness", "Food", "Location", "Rooms", "Service"]
        if dataset_name == "amasum_shoes":
            facets = ["Breathability", "Durability", "Weight", "Cushioning", "Stability", "Flexibility", "Traction",
                      "Sizefit", "Comfort", "Misc"]

        generations_info = info[dataset_name]

        # all generation results share the same review-categorization
        generation_file_target = ""
        for generation_info in generations_info:
            if generation_info["model_name"] == "modular_llama3":
                generation_file_target = generation_info["generation_file"]
        categorization_file = "_".join(generation_file_target.split("/")[1:]).split(".")[0] + ".json"
        with open("../eval_auto/categorization/" + categorization_file) as f:
            samples = json.load(f)
        source_texts = []
        for sample in samples:
            source_texts.append("\n".join(sample["source_documents"]))


        # human reference
        generation_file = generations_info[0]["generation_file"]
        print("human reference")
        candidate_key = generations_info[0]["candidate_key"]
        reference_key = generations_info[0]["reference_key"]

        categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
        with open("../eval_auto/categorization/" + categorization_file) as f:
            samples = json.load(f)

        references = []
        reference_categorizations = []
        for sample in samples:
            if isinstance(sample[reference_key], str):
                references.append(sample[reference_key])
            else:
                references.append(sample[reference_key][0])  # SPACE has multiple references
            reference_categorizations.append(sample["categorization_reference"])
        analyzing(references, source_texts)

        # model generations
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]

            # use the processed result with shared content from categorization
            categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
            with open("../eval_auto/categorization/" + categorization_file) as f:
                samples = json.load(f)

            candidates = []
            candidate_categorizations = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                candidate_categorizations.append(sample["categorization_candidate"])

            analyzing(candidates, source_texts)