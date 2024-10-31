import json
import numpy as np

# print("space")
# with open("space_alignscore_full.json") as f:
#     results = json.load(f)
#
# for model, result in results.items():
#     print(model)
#     print(np.mean(result))
#
# print("peermeta")
# with open("peermeta_alignscore_full.json") as f:
#     results = json.load(f)
#
# for model, result in results.items():
#     print(model)
#     print(np.mean(result))

with open("all.json") as f:
    info = json.load(f)

dataset_names = ["space", "peermeta", "amasum_shoes"]
# dataset_names = ["amasum_shoes"]
for dataset_name in dataset_names:
    print(dataset_name)
    generations_info = info[dataset_name]
    for generation_info in generations_info:
        generation_file = generation_info["generation_file"]
        with open(generation_file) as f:
            samples = json.load(f)
        print(len(samples), generation_file)