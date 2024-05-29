import os
import urllib
import jsonlines


conference = "iclr_2018"
pdf_folder = f"../data/pdfs_{conference}"

existing_ids = []
for file in os.listdir(pdf_folder):
    existing_ids.append(file[:-4])

with jsonlines.open(f"../data/{conference}.jsonl") as reader:
    for line in reader:
        id = line["id"]
        if id not in existing_ids:
            url = line["pdf"]
            urllib.urlretrieve(url, f"{pdf_folder}/{id}.pdf")

