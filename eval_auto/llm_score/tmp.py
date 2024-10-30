import json
import numpy as np

with open("space_llm_scored_full_tmp.json") as f:
    results = json.load(f)

for model, result in results.items():
    print(model)
    print(np.mean(result))