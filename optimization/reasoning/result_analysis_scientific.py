import json
import numpy as np


with open("scientific_reasoning_result_gpt_4o.json") as f:
    samples = json.load(f)

sources = []
ground_truths = []
generations = []
for key in samples.keys():
    sample = samples[key]
    categorization_pairs = sample["categorization_pairs"]
    for pair in categorization_pairs:
        review_fargments = pair["review_fragments"]
        meta_review_fragments = pair["meta_review_fragments"]
        if len(review_fargments) > 0 and len(meta_review_fragments) > 0:
            sources.append(" ".join(review_fargments))
            ground_truths.append(" ".join(meta_review_fragments))
            generations.append(pair["meta_generated"])

import sys
sys.path.append("../../")
from utils.metrics import evaluating_summaries_single_source
print(evaluating_summaries_single_source(ground_truths, generations, sources))


from utils.metrics import summac_scores
scores_zs, scores_conv = summac_scores(sources, generations)
print("scores zs generation", np.mean(scores_zs), "scores conv generation", np.mean(scores_conv))
scores_zs, scores_conv = summac_scores(sources, ground_truths)
print("scores zs ground truths", np.mean(scores_zs), "scores conv ground truth", np.mean(scores_conv))


with open("scientific_reasoning_result_llama31_70b.json") as f:
    samples = json.load(f)

sources = []
ground_truths = []
generations = []
for key in samples.keys():
    sample = samples[key]
    categorization_pairs = sample["categorization_pairs"]
    for pair in categorization_pairs:
        review_fargments = pair["review_fragments"]
        meta_review_fragments = pair["meta_review_fragments"]
        meta_generated = pair["meta_generated"]
        if len(review_fargments) > 0 and len(meta_review_fragments) > 0 and meta_generated.strip() != "":
            sources.append(" ".join(review_fargments))
            ground_truths.append(" ".join(meta_review_fragments))
            generations.append(meta_generated)

print(len(sources), len(ground_truths), len(generations))
import sys
sys.path.append("../../")
from utils.metrics import evaluating_summaries_single_source
print(evaluating_summaries_single_source(ground_truths, generations, sources))


from utils.metrics import summac_scores
scores_zs, scores_conv = summac_scores(sources, generations)
print("scores zs generation", np.mean(scores_zs), "scores conv generation", np.mean(scores_conv))
scores_zs, scores_conv = summac_scores(sources, ground_truths)
print("scores zs ground truths", np.mean(scores_zs), "scores conv ground truth", np.mean(scores_conv))