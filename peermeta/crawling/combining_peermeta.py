import jsonlines

iclr_2018 = []
with jsonlines.open("../data/iclr_2018.jsonl") as reader:
    for line in reader:
        iclr_2018.append(line)
print(len(iclr_2018))
print(iclr_2018[0])

iclr_2019 = []
with jsonlines.open("../data/iclr_2019.jsonl") as reader:
    for line in reader:
        iclr_2019.append(line)
print(len(iclr_2019))
print(iclr_2019[0])

iclr_2020 = []
with jsonlines.open("../data/iclr_2020.jsonl") as reader:
    for line in reader:
        iclr_2020.append(line)
print(len(iclr_2020))
print(iclr_2020[0])

iclr_2021 = []
with jsonlines.open("../data/iclr_2021.jsonl") as reader:
    for line in reader:
        iclr_2021.append(line)
print(len(iclr_2021))
print(iclr_2021[0])

iclr_2022 = []
with jsonlines.open("../data/iclr_2022.jsonl") as reader:
    for line in reader:
        iclr_2022.append(line)
print(len(iclr_2022))
print(iclr_2022[0])

iclr_2023 = []
with jsonlines.open("../data/iclr_2023.jsonl") as reader:
    for line in reader:
        iclr_2023.append(line)
print(len(iclr_2023))
print(iclr_2023[0])

iclr_2024 = []
with jsonlines.open("../data/iclr_2024.jsonl") as reader:
    for line in reader:
        iclr_2024.append(line)
print(len(iclr_2024))
print(iclr_2024[0])


nips_2021 = []
with jsonlines.open("../data/nips_2021.jsonl") as reader:
    for line in reader:
        nips_2021.append(line)
print(len(nips_2021))
print(nips_2021[0])

nips_2022 = []
with jsonlines.open("../data/nips_2022.jsonl") as reader:
    for line in reader:
        nips_2022.append(line)
print(len(nips_2022))
print(nips_2022[0])

nips_2023 = []
with jsonlines.open("../data/nips_2023.jsonl") as reader:
    for line in reader:
        nips_2023.append(line)
print(len(nips_2023))
print(nips_2023[0])
