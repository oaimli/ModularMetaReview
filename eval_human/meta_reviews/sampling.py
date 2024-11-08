import json
import random

random.seed(42)

with open("info.json") as f:
    all_info = json.load(f)

for dataset_name, generation_infos in all_info.items():
    print(dataset_name)
    samples = {}
    for generation_info in generation_infos:
        candidate_key = generation_info["candidate_key"]
        reference_key = generation_info["reference_key"]
        with open(generation_info["generation_file"]) as f:
            generations = json.load(f)
        for i, generation in enumerate(generations):
            sample_new = samples.get(f"index_{i}", {})
            sample_new["source_documents"] = generation["source_documents"]
            tmp = sample_new.get("generations", {})
            tmp["human_reference"]: generation[reference_key]
            tmp[generation_info["model_name"]] = generation[candidate_key]
            sample_new["generations"] = tmp
            samples[f"index_{i}"] = sample_new

    indexes = []
    for sample_key, sample_value in samples.items():
        source_documents = sample_value["source_documents"]
        if dataset_name == "peermeta":
            source_threshhold = 8
        elif dataset_name == "amasum_shoes":
            source_threshhold = 300
        elif dataset_name == "space":
            source_threshhold = 80
        else:
            print("the dataset name is incorrect.")
            source_threshhold = 0

        if len(source_documents) <= source_threshhold:
            indexes.append(sample_key)
    print(len(indexes))

    with open(f"generations_{dataset_name}.json", "w") as f:
        json.dump(samples, f, indent=4)

