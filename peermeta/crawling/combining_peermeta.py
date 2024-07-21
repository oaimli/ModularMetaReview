import random

import jsonlines


samples_all = []

iclr_2018 = []
with jsonlines.open("../data/iclr_2018.jsonl") as reader:
    for line in reader:
        iclr_2018.append(line)
print(len(iclr_2018))
print(iclr_2018[0])
samples_all.extend(iclr_2018)

iclr_2019 = []
with jsonlines.open("../data/iclr_2019.jsonl") as reader:
    for line in reader:
        iclr_2019.append(line)
print(len(iclr_2019))
print(iclr_2019[0])
samples_all.extend(iclr_2019)

iclr_2020 = []
with jsonlines.open("../data/iclr_2020.jsonl") as reader:
    for line in reader:
        iclr_2020.append(line)
print(len(iclr_2020))
print(iclr_2020[0])
samples_all.extend(iclr_2020)

iclr_2021 = []
with jsonlines.open("../data/iclr_2021.jsonl") as reader:
    for line in reader:
        iclr_2021.append(line)
print(len(iclr_2021))
print(iclr_2021[0])
samples_all.extend(iclr_2021)

iclr_2022 = []
with jsonlines.open("../data/iclr_2022.jsonl") as reader:
    for line in reader:
        iclr_2022.append(line)
print(len(iclr_2022))
print(iclr_2022[0])
samples_all.extend(iclr_2022)

iclr_2023 = []
with jsonlines.open("../data/iclr_2023.jsonl") as reader:
    for line in reader:
        iclr_2023.append(line)
print(len(iclr_2023))
print(iclr_2023[0])
samples_all.extend(iclr_2023)

iclr_2024 = []
with jsonlines.open("../data/iclr_2024.jsonl") as reader:
    for line in reader:
        iclr_2024.append(line)
print(len(iclr_2024))
print(iclr_2024[0])
samples_all.extend(iclr_2024)


nips_2021 = []
with jsonlines.open("../data/nips_2021.jsonl") as reader:
    for line in reader:
        nips_2021.append(line)
print(len(nips_2021))
print(nips_2021[0])
samples_all.extend(nips_2021)

nips_2022 = []
with jsonlines.open("../data/nips_2022.jsonl") as reader:
    for line in reader:
        nips_2022.append(line)
print(len(nips_2022))
print(nips_2022[0])
samples_all.extend(nips_2022)

nips_2023 = []
with jsonlines.open("../data/nips_2023.jsonl") as reader:
    for line in reader:
        nips_2023.append(line)
print(len(nips_2023))
print(nips_2023[0])
samples_all.extend(nips_2023)


random.seed(42)
all_num = len(samples_all)
train_num = int(all_num * 0.8)
val_num = int(all_num * 0.1)

papers_indexes = range(all_num)
papers_train_indexes = random.sample(papers_indexes, train_num)
for i in papers_train_indexes:
    samples_all[i]["label"] = "train"
print("train", len(papers_train_indexes))

papers_val_test_indexes = [item for item in papers_indexes if item not in papers_train_indexes]
papers_val_indexes = random.sample(papers_val_test_indexes, val_num)
for i in papers_val_indexes:
    samples_all[i]["label"] = "val"
print("val", len(papers_val_indexes))

papers_test_indexes = [item for item in papers_val_test_indexes if item not in papers_val_indexes]
for i in papers_test_indexes:
    samples_all[i]["label"] = "test"
print("test", len(papers_test_indexes))

with jsonlines.open("../data/peermeta_all.jsonl", "w") as writer:
    writer.write_all(samples_all)


