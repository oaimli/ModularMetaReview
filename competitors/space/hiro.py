"""
Get generation results from the github of HIRO,
HIRO: Hierarchical Indexing for Retrieval-Augmented Opinion Summarization
As the orders in the two test sets are totally matched, we directly obtain results from the repository
"""
import json

import jsonlines

test_samples = []
# read our test data
with jsonlines.open("../../datasets/space_test.jsonl") as reader:
    for line in reader:
        test_samples.append(line)

# read the generation results from the github repository
generations_extractive = []
with open("../../../tmp/hiro-main/output/space/hiro_extractive_test.txt") as f:
    for line in f.readlines():
        generations_extractive.append(line)
results_extractive = []
for sample, output in zip(test_samples, generations_extractive):
    sample["generated_summary_general"] = output
    results_extractive.append(sample)
with open("../../results/hiro_space/hiro_extractive.json", "w") as f:
    json.dump(results_extractive, f)


generations_abstractive_one = []
with open("../../../tmp/hiro-main/output/space/hiro_oneshot_test_llama7b_1.txt") as f:
    for line in f.readlines():
        generations_abstractive_one.append(line)
results_abstractive_one = []
for sample, output in zip(test_samples, generations_abstractive_one):
    sample["generated_summary_general"] = output
    results_abstractive_one.append(sample)
with open("../../results/hiro_space/hiro_oneshot_llama7b_1.json", "w") as f:
    json.dump(results_abstractive_one, f)

generations_abstractive_two = []
with open("../../../tmp/hiro-main/output/space/hiro_oneshot_test_llama7b_2.txt") as f:
    for line in f.readlines():
        generations_abstractive_two.append(line)
results_abstractive_two = []
for sample, output in zip(test_samples, generations_abstractive_two):
    sample["generated_summary_general"] = output
    results_abstractive_two.append(sample)
with open("../../results/hiro_space/hiro_oneshot_llama7b_2.json", "w") as f:
    json.dump(results_abstractive_two, f)

generations_abstractive_three = []
with open("../../../tmp/hiro-main/output/space/hiro_oneshot_test_llama7b_3.txt") as f:
    for line in f.readlines():
        generations_abstractive_three.append(line)
results_abstractive_three = []
for sample, output in zip(test_samples, generations_abstractive_three):
    sample["generated_summary_general"] = output
    results_abstractive_three.append(sample)
with open("../../results/hiro_space/hiro_oneshot_llama7b_3.json", "w") as f:
    json.dump(results_abstractive_three, f)