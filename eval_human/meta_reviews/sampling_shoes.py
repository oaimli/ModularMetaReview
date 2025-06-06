import json
import random

random.seed(42)

with open("info.json") as f:
    all_info = json.load(f)

dataset_name = "amasum_shoes"
generation_infos = all_info[dataset_name]
print(dataset_name)
samples = {}
for generation_info in generation_infos:
    print(generation_info)
    candidate_key = generation_info["candidate_key"]
    reference_key = generation_info["reference_key"]
    with open(generation_info["generation_file"]) as f:
        generations = json.load(f)
    for i, generation in enumerate(generations):
        sample_new = samples.get(f"index_{i}", {})
        source_documents = sample_new.get("source_documents", [])
        if len(source_documents) > 0:
            if source_documents[0] == generation["source_documents"][0]:
                print("source documents are consistent")
        else:
            print("there have been no source documents yet")
        sample_new["source_documents"] = generation["source_documents"]
        tmp = sample_new.get("generations", {})
        if dataset_name == "space":
            tmp["human_reference"] = generation[reference_key][random.randint(0, 2)]
        else:
            tmp["human_reference"] = generation[reference_key]

        other_references = []
        for generation_tmp in generations:
            if dataset_name == "space":
                reference_tmp = generation_tmp[reference_key][random.randint(0, 2)]
            else:
                reference_tmp = generation_tmp[reference_key]
            if reference_tmp != tmp["human_reference"]:
                other_references.append(reference_tmp)
        tmp["random_reference"] = random.choice(other_references)

        tmp[generation_info["model_name"]] = generation[candidate_key]
        sample_new["generations"] = tmp
        samples[f"index_{i}"] = sample_new

indexes = []
for sample_key, sample_value in samples.items():
    source_documents = sample_value["source_documents"]
    source_text_length = len("\n".join(source_documents).split())
    if len(source_documents) <= 200 and source_text_length <= 5000:
        indexes.append(sample_key)
print(len(indexes))
print(indexes)

# indexes_sampled = random.sample(indexes, 10)
indexes_sampled = ['index_43', 'index_31', 'index_22', 'index_29', 'index_4', 'index_48', 'index_42', 'index_24', 'index_13', 'index_8']
print(indexes_sampled)

samples_sampled = {}
for sample_index in indexes_sampled:
    samples_sampled[sample_index] = samples[sample_index]

with open(f"generations_{dataset_name}.json", "w") as f:
    json.dump(samples_sampled, f, indent=4)
