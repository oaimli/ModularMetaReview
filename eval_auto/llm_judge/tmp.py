import  json
import jsonlines


with open("../all.json") as f:
    info = json.load(f)

dataset_names = ["peermeta", "space", "amasum_shoes"]
for dataset_name in dataset_names:
    print(dataset_name)
    all_samples = []
    with jsonlines.open(f"../../datasets/{dataset_name}_test.jsonl") as reader:
        for line in reader:
            all_samples.append(line)

    generation_files = info[dataset_name]
    for generation_file in generation_files:
        print(generation_file)
        with open(generation_file["generation_file"]) as f:
            results = json.load(f)
        print(len(results))

output = ["a", "a", "b"]
prediction = max(output, key=output.count)
print(prediction)
