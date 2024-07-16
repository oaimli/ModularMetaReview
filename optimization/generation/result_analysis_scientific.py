import json
import numpy as np

with open("scientific_generation_result_gpt4.json") as f:
    samples = json.load(f)

sources = []
ground_truths = []
generations = []
for key in samples.keys():
    sample = samples[key]

    reviews = []
    for r in sample["reviews"]:
        reviews.append(r["comment"])
    sources.append("\n".join(reviews))
    ground_truths.append(sample["meta_review"])
    generations.append(sample["meta_review_generated"])

import sys
sys.path.append("../../")
from utils.metrics import evaluating_summaries_single_source
print(evaluating_summaries_single_source(ground_truths, generations, sources))


from utils.metrics import summac_scores
scores_zs, scores_conv = summac_scores(sources, generations)
print("scores zs generation", np.mean(scores_zs), "scores conv generation", np.mean(scores_conv))
scores_zs, scores_conv = summac_scores(sources, ground_truths)
print("scores zs ground truths", np.mean(scores_zs), "scores conv ground truth", np.mean(scores_conv))