import json


with open("../../../optimization/organization/scientific_categorization_result_llama31_70b_processed.json") as f:
    processed_samples = json.load(f)

with open("../../../annotations/scientific_reviews/ze_annotation_result_fragments.json") as f:
    zenan_results = json.load(f)

shared_ids = list(
    set(zenan_results.keys()).intersection(set(processed_samples.keys())))
print(len(shared_ids))

for key in shared_ids:
    sample = processed_samples[key]
    sample_human = zenan_results[key]

    meta_review_categorization = sample["meta_review_categorization"]
    review_categorizations = sample_human["review_categorization"]

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