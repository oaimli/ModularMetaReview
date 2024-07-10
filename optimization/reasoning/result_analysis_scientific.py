import json

with open("scientific_reasoning_result_gpt4.json") as f:
    samples = json.load(f)

inputs = []
ground_truths = []
generations = []
for key in samples.keys():
