import json


with open("../organization/scientific_categorization_result_llama31_70b_processed.json") as f:
    processed_samples = json.load(f)

for key in processed_samples.keys():
    sample = processed_samples[key]

    meta_review_categorization = sample["meta_review_categorization"]
    review_categorizations = sample["review_categorization"]

    categorization_pairs = []
    for facet in meta_review_categorization.keys():
        tmp = {}
        tmp["facet"] = facet

        review_fragments = []
        for review_categorization in review_categorizations:
            review_fragments.extend(review_categorization[facet])
        tmp["review_fragments"] = review_fragments

        tmp["meta_review_fragments"] = meta_review_categorization[facet]

        categorization_pairs.append(tmp)

    processed_samples[key]["categorization_pairs"] = categorization_pairs

with open("scientific_selection_result_llama31_70b.json", "w") as f:
    json.dump(processed_samples, f, indent=4)