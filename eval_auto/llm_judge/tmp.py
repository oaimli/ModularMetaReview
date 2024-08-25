import  json

with open("../info.json") as f:
    info = json.load(f)

dataset_names = ["peermeta", "space", "amasum_shoes"]
for dataset_name in dataset_names:
    print(dataset_name)
    generation_files = info[dataset_name]
    for generation_file in generation_files:
        print(generation_file)
        with open(generation_file["generation_file"]) as f:
            print(len(json.load(f)))