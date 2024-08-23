# This is just to group extracted fragments into different review facets
import json


model_name = "llama31_70b"
with open(f"peermeta_categorization_result_{model_name}.json") as f:
    processed_samples = json.load(f)

for sample_index, sample in enumerate(processed_samples):
    review_categorizations = sample["review_categorization"]

    categorization_pairs = []
    for facet in review_categorizations[0].keys():
        tmp = {}
        tmp["facet"] = facet

        review_fragments = []
        for review_categorization in review_categorizations:
            review_fragments.extend(review_categorization[facet])
        tmp["review_fragments"] = review_fragments
        categorization_pairs.append(tmp)

    processed_samples[sample_index]["categorization_pairs"] = categorization_pairs

with open(f"peermeta_selection_result_{model_name}.json", "w") as f:
    json.dump(processed_samples, f, indent=4)