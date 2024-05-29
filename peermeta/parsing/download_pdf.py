import os
import wget
import jsonlines
from tqdm import tqdm


conference = "iclr_2018"
pdf_folder = f"../data/pdfs_{conference}"

existing_ids = []
for file in os.listdir(pdf_folder):
    existing_ids.append(file[:-4])
print(existing_ids)

pdfs = {}
with jsonlines.open(f"../data/{conference}.jsonl") as reader:
    for line in reader:
        pdfs[line["id"]] = line["pdf"]

for id in tqdm(pdfs.keys()):
    if id not in existing_ids:
        url = pdfs[id]
        wget.download(url, out=f"{pdf_folder}/{id}.pdf")

