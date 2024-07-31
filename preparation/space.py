import json
import os.path

space_folder = "../datasets/space/data"
splits = {}
with open(os.path.join(space_folder, "space_summ_splits.txt")) as f:
    for line in f.readlines():
        tmp = line.strip().split()
        splits[tmp[0]] = tmp[1]

print(splits)

# split to dev and test
with open(os.path.join(space_folder, "json/space_summ.json")) as f:
    sources = json.load(f)
print(len(sources)) # 50
for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["label"] = splits[entity_id]
    del sources[i]["summaries"]

# building
folder = os.path.join(space_folder, "gold/building")
files = os.listdir(folder)
gold_summaries = {}
for file in files:
    print(file)
    entity_id = file[:-6]
    order = file[-5]
    with open(os.path.join(folder, file)) as f:
        sentences = f.read().split("\t")
        print(len(sentences))
        gold_summaries[entity_id] = gold_summaries.get(entity_id, {})
        gold_summaries[entity_id][order] = sentences

for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["gold_summaries_building"] = gold_summaries[entity_id]

# cleanliness
folder = os.path.join(space_folder, "gold/cleanliness")
files = os.listdir(folder)
gold_summaries = {}
for file in files:
    print(file)
    entity_id = file[:-6]
    order = file[-5]
    with open(os.path.join(folder, file)) as f:
        sentences = f.read().split("\t")
        print(len(sentences))
        gold_summaries[entity_id] = gold_summaries.get(entity_id, {})
        gold_summaries[entity_id][order] = sentences

for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["gold_summaries_cleanliness"] = gold_summaries[entity_id]

# food
folder = os.path.join(space_folder, "gold/food")
files = os.listdir(folder)
gold_summaries = {}
for file in files:
    print(file)
    entity_id = file[:-6]
    order = file[-5]
    with open(os.path.join(folder, file)) as f:
        sentences = f.read().split("\t")
        print(len(sentences))
        gold_summaries[entity_id] = gold_summaries.get(entity_id, {})
        gold_summaries[entity_id][order] = sentences

for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["gold_summaries_food"] = gold_summaries[entity_id]

# location
folder = os.path.join(space_folder, "gold/location")
files = os.listdir(folder)
gold_summaries = {}
for file in files:
    print(file)
    entity_id = file[:-6]
    order = file[-5]
    with open(os.path.join(folder, file)) as f:
        sentences = f.read().split("\t")
        print(len(sentences))
        gold_summaries[entity_id] = gold_summaries.get(entity_id, {})
        gold_summaries[entity_id][order] = sentences

for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["gold_summaries_location"] = gold_summaries[entity_id]

# rooms
folder = os.path.join(space_folder, "gold/rooms")
files = os.listdir(folder)
gold_summaries = {}
for file in files:
    print(file)
    entity_id = file[:-6]
    order = file[-5]
    with open(os.path.join(folder, file)) as f:
        sentences = f.read().split("\t")
        print(len(sentences))
        gold_summaries[entity_id] = gold_summaries.get(entity_id, {})
        gold_summaries[entity_id][order] = sentences

for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["gold_summaries_rooms"] = gold_summaries[entity_id]

# cleanliness
folder = os.path.join(space_folder, "gold/service")
files = os.listdir(folder)
gold_summaries = {}
for file in files:
    print(file)
    entity_id = file[:-6]
    order = file[-5]
    with open(os.path.join(folder, file)) as f:
        sentences = f.read().split("\t")
        print(len(sentences))
        gold_summaries[entity_id] = gold_summaries.get(entity_id, {})
        gold_summaries[entity_id][order] = sentences

for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["gold_summaries_service"] = gold_summaries[entity_id]

# general
folder = os.path.join(space_folder, "gold/general")
files = os.listdir(folder)
gold_summaries = {}
for file in files:
    print(file)
    entity_id = file[:-6]
    order = file[-5]
    with open(os.path.join(folder, file)) as f:
        sentences = f.read().split("\t")
        print(len(sentences))
        gold_summaries[entity_id] = gold_summaries.get(entity_id, {})
        gold_summaries[entity_id][order] = sentences

for i, source in enumerate(sources):
    entity_id = source["entity_id"]
    sources[i]["gold_summaries_general"] = gold_summaries[entity_id]


samples_dev = []
samples_test = []
for source in sources:
    if source["label"] == "dev":
        samples_dev.append(source)
    elif source["label"] == "test":
        samples_test.append(source)
    else:
        print("Errors in the label names")

print(len(samples_test), len(samples_dev))

with open("../datasets/space_dev.json", "w") as f:
    json.dump(samples_dev, f, indent=4)

with open("../datasets/space_test.json", "w") as f:
    json.dump(samples_test, f, indent=4)