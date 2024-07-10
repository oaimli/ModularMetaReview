import json


with open("scientific_selection_result_llama3_70b.json") as f:
    samples = json.load(f)

all_pairs = []
invalid_pairs = []
for key in samples.keys():
    sample = samples[key]
    categorization_pairs = sample["categorization_pairs"]
    for pair in categorization_pairs:
        all_pairs.append(pair)
        review_fargments = pair["review_fragments"]
        meta_review_fragments = pair["meta_review_fragments"]
        if len(review_fargments) == 0 or len(meta_review_fragments) == 0:
            invalid_pairs.append(pair)

print("Ratio of invalid pairs", len(invalid_pairs)/len(all_pairs))
# Ratio of invalid pairs 0.3