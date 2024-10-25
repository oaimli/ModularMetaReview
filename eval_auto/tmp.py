import json
import numpy as np

print("space")
with open("space_alignscore_full.json") as f:
    results = json.load(f)

for model, result in results.items():
    print(model)
    print(np.mean(result))

print("peermeta")
with open("peermeta_alignscore_full.json") as f:
    results = json.load(f)

for model, result in results.items():
    print(model)
    print(np.mean(result))