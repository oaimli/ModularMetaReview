import numpy as np
import json
from openai import OpenAI


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    with open("info.json") as f:
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
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]

            # use the processed result with shared content from categorization
            categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
            with open(categorization_file) as f:
                samples = json.load(f)

            candidates = []
            references = []
            review_categorizations = []
            candidate_categorizations = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                if isinstance(sample[reference_key], str):
                    references.append(sample[reference_key])
                else:
                    references.append(sample[reference_key][0])  # SPACE has multiple references
                review_categorizations.append(sample["review_categorization"])
                candidate_categorizations.append(sample["categorization_candidate"])

            # compared with references on only shared aspects
            recalls = []
            precisions = []
            f_measures = []
            for review_categorization, candidate_categorization in zip(review_categorizations, candidate_categorizations):
                review_count = 0
                shared_count = 0
                candidate_count = 0
                for facet in facets:
                    flag = 0
                    for categorization in review_categorization:
                        if len(categorization[facet]) > 0:
                            flag = 1
                            break
                    if flag == 1:
                        review_count += 1

                    if len(candidate_categorization[facet]) > 0:
                        candidate_count += 0

                    if flag == 1 and len(candidate_categorization[facet]) > 0:
                        shared_count += 1
                r = (shared_count + 1) / (review_count + 1)
                p = (shared_count + 1) / (candidate_count + 1)
                recalls.append(r)
                precisions.append(p)
                f_measures.append(2 * (r * p) / (r + p))
            print("recall", np.mean(recalls), "precision", np.mean(precisions), "f-measure", np.mean(f_measures))
