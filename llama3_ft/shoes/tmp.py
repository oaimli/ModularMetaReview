import jsonlines
import json

results = []
with jsonlines.open("/home/miao4/punim0521/ModularMetaReview/results/llama31_8b_amasum_shoes/generated_summaries.jsonl") as reader:
    for line in reader:
        results.append(line)

with open("/home/miao4/punim0521/ModularMetaReview/results/llama31_8b_amasum_shoes/generated_summaries.json", "w") as f:
    json.dump(results, f)