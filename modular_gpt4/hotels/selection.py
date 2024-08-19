import json


with open("shoes_categorization_result_llama3_70b_processed.json") as f:
    processed_samples = json.load(f)

for key in processed_samples.keys():
    sample = processed_samples[key]
    review_categorizations = sample["review_categorization"]

    categorization_pairs = []
    for facet in meta_review_categorization.keys():
        tmp = {}
        tmp["facet"] = facet

        review_fargments = []
        for review_categorization in review_categorizations:
            review_fargments.extend(review_categorization[facet])
        tmp["review_fragments"] = review_fargments

        tmp["meta_review_fragments"] = meta_review_categorization[facet]

        categorization_pairs.append(tmp)

    processed_samples[key]["categorization_pairs"] = categorization_pairs

with open("scientific_selection_result_llama3_70b.json", "w") as f:
    json.dump(processed_samples, f, indent=4)