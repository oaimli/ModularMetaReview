import json


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
        review_categorizations = []
        for sample in samples:
            review_categorizations.append(sample["review_categorization"])
        source_distribution = {}
        for review_categorization in review_categorizations:
            for categorization in review_categorization:
                for facet, fragments in categorization.items():
                    if len(fragments) > 0:
                        source_distribution[facet] = source_distribution.get(facet, 0) + 1
        source_distribution_ratio = {}
        for facet, count in source_distribution.items():
            source_distribution_ratio[facet] = count / sum(list(source_distribution.values()))
        print("source_distribution_ratio", sorted(source_distribution_ratio.items(), key = lambda i: i[0]))

        # human reference
        generation_file = generations_info[0]["generation_file"]
        print("human reference")
        candidate_key = generations_info[0]["candidate_key"]
        reference_key = generations_info[0]["reference_key"]

        categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
        with open("../eval_auto/categorization/" + categorization_file) as f:
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

        reference_distribution = {}
        for reference_categorization in reference_categorizations:
            for facet, fragments in reference_categorization.items():
                if len(fragments) > 0:
                    reference_distribution[facet] = reference_distribution.get(facet, 0) + 1
        reference_distribution_ratio = {}
        for facet, count in reference_distribution.items():
            reference_distribution_ratio[facet] = count / sum(list(reference_distribution.values()))
        print("reference_distribution_ratio", sorted(reference_distribution_ratio.items(), key = lambda i: i[0]))

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
            references = []
            candidate_categorizations = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                if isinstance(sample[reference_key], str):
                    references.append(sample[reference_key])
                else:
                    references.append(sample[reference_key][0])  # SPACE has multiple references
                candidate_categorizations.append(sample["categorization_candidate"])

            candidate_distribution = {}
            for candidate_categorization in candidate_categorizations:
                for facet, fragments in candidate_categorization.items():
                    if len(fragments) > 0:
                        candidate_distribution[facet] = candidate_distribution.get(facet, 0) + 1
            candidate_distribution_ratio = {}
            for facet, count in candidate_distribution.items():
                candidate_distribution_ratio[facet] = count / sum(list(candidate_distribution.values()))
            print("candidate_distribution_ratio", sorted(candidate_distribution_ratio.items(), key = lambda i: i[0]))