"""
Get generation results from the github of HIRO,
HIRO: Hierarchical Indexing for Retrieval-Augmented Opinion Summarization
"""
import jsonlines
import json

# read our test data for the sports shoes domain
test_samples_ours = []
with jsonlines.open("../../datasets/amasum_shoes_test.jsonl") as reader:
    for line in reader:
        test_samples_ours.append(line)

# read the test set from the github repository, there are 200 test samples for four domains
test_samples_hiro = []
with jsonlines.open("../../../tmp/hiro-main/data/amasum/opagg/amasum-eval-combined/test.jsonl") as reader:
    for line in reader:
        test_samples_hiro.append(line)

selected_indexes = []
for sample_ours in test_samples_ours:
    entity_id = sample_ours["entity_id"]
    for i, sample_hiro in enumerate(test_samples_hiro):
        if sample_hiro["entity_id"] == entity_id:
            selected_indexes.append(i)
            break
print(selected_indexes, len(selected_indexes)) # index 50-99

# read the generation results from the github repository, and extract samples for the shoes domain
generations_extractive = []
with open("../../../tmp/hiro-main/output/amasum-combined/hiro_extractive_test.txt") as f:
    for line in f.readlines():
        generations_extractive.append(line)
results_extractive = []
for sample, output in zip(test_samples_ours, generations_extractive[50:100]):
    sample["generated_meta_review"] = output
    results_extractive.append(sample)
with open("../../results/hiro_amasum_shoes/hiro_extractive.json", "w") as f:
    json.dump(results_extractive, f)

generations_abstractive_one = []
with open("../../../tmp/hiro-main/output/amasum-combined/hiro_piecewise_test_llama7b_1.txt") as f:
    for line in f.readlines():
        generations_abstractive_one.append(line)
results_abstractive_one = []
for sample, output in zip(test_samples_ours, generations_abstractive_one[50:100]):
    sample["generated_meta_review"] = output
    results_abstractive_one.append(sample)
with open("../../results/hiro_amasum_shoes/hiro_piecewise_llama7b_1.json", "w") as f:
    json.dump(results_abstractive_one, f)


generations_abstractive_two = []
with open("../../../tmp/hiro-main/output/amasum-combined/hiro_piecewise_test_llama7b_2.txt") as f:
    for line in f.readlines():
        generations_abstractive_two.append(line)
results_abstractive_two = []
for sample, output in zip(test_samples_ours, generations_abstractive_two[50:100]):
    sample["generated_meta_review"] = output
    results_abstractive_two.append(sample)
with open("../../results/hiro_amasum_shoes/hiro_piecewise_llama7b_2.json", "w") as f:
    json.dump(results_abstractive_two, f)


generations_abstractive_three = []
with open("../../../tmp/hiro-main/output/amasum-combined/hiro_piecewise_test_llama7b_3.txt") as f:
    for line in f.readlines():
        generations_abstractive_three.append(line)
results_abstractive_three = []
for sample, output in zip(test_samples_ours, generations_abstractive_three[50:100]):
    sample["generated_meta_review"] = output
    results_abstractive_three.append(sample)
with open("../../results/hiro_amasum_shoes/hiro_piecewise_llama7b_3.json", "w") as f:
    json.dump(results_abstractive_three, f)