import numpy as np
import json


if __name__ == "__main__":
    with open("all.json") as f:
        info = json.load(f)

    # dataset_names = ["peermeta", "space", "amasum_shoes"]
    dataset_names = ["amasum_shoes"]
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

        # all generation results share the same review-categorization and there must be the modular row in the batch
        generation_file_target = ""
        for generation_info in generations_info:
            if generation_info["model_name"] == "modular_llama3":
                generation_file_target = generation_info["generation_file"]
                generation_file_target = generation_file_target[:-5] + "_truncated.json"
        categorization_file = "_".join(generation_file_target.split("/")[1:]).split(".")[0] + ".json"
        with open("categorization/" + categorization_file) as f:
            samples = json.load(f)
        review_categorizations = []
        for sample in samples:
            review_categorizations.append(sample["review_categorization"])


        # human reference
        generation_file = generations_info[0]["generation_file"]
        generation_file = generation_file[:-5] + "_truncated.json"
        print("human reference")
        candidate_key = generations_info[0]["candidate_key"]
        reference_key = generations_info[0]["reference_key"]

        categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
        with open("categorization/" + categorization_file) as f:
            samples = json.load(f)

        candidates = []
        references = []
        reference_categorizations = []
        for sample in samples:
            candidates.append(sample[candidate_key])
            if isinstance(sample[reference_key], str):
                references.append(sample[reference_key])
            else:
                references.append(sample[reference_key][0])  # SPACE has multiple references
            reference_categorizations.append(sample["categorization_reference"])

        recalls = []
        precisions = []
        f_measures = []
        for review_categorization, reference_categorization in zip(review_categorizations, reference_categorizations):
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

                if len(reference_categorization[facet]) > 0:
                    candidate_count += 1

                if flag == 1 and len(reference_categorization[facet]) > 0:
                    shared_count += 1
            r = (shared_count + 1) / (review_count + 1)
            p = (shared_count + 1) / (candidate_count + 1)
            recalls.append(r)
            precisions.append(p)
            f_measures.append(2 * (r * p) / (r + p))
        print("recall", np.mean(recalls), "precision", np.mean(precisions), "f-measure", np.mean(f_measures))

        # model generations
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            generation_file = generation_file[:-5] + "_truncated.json"
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]

            # use the processed result with shared content from categorization
            categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
            with open("categorization/" + categorization_file) as f:
                samples = json.load(f)

            candidates = []
            references = []
            candidate_categorizations = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                if isinstance(sample[reference_key], str):
                    references.append(sample[reference_key])
                else:
                    references.append(sample[reference_key][0])  # SPACE has multiple references
                candidate_categorizations.append(sample["categorization_candidate"])

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
                        candidate_count += 1

                    if flag == 1 and len(candidate_categorization[facet]) > 0:
                        shared_count += 1
                r = (shared_count + 1) / (review_count + 1)
                p = (shared_count + 1) / (candidate_count + 1)
                recalls.append(r)
                precisions.append(p)
                f_measures.append(2 * (r * p) / (r + p))
            print("recall", np.mean(recalls), "precision", np.mean(precisions), "f-measure", np.mean(f_measures))
