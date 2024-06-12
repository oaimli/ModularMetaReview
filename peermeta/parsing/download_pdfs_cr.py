# Download the pdfs of the final versions
# camera ready versions for accepted papers and submission versions for rejected papers
import os
import wget
import jsonlines
from tqdm import tqdm
from urllib.error import HTTPError


conference = "iclr_2023"
print(conference)
pdf_folder = f"../data/pdfs_cr_{conference}"

existing_ids = []
for file in os.listdir(pdf_folder):
    existing_ids.append(file[:-4])
print(existing_ids)

pdfs = {}
with jsonlines.open(f"../data/{conference}.jsonl") as reader:
    for line in reader:
        id = line["id"]
        if id not in existing_ids:
            pdfs[line["id"]] = line["pdf"]

for id in tqdm(pdfs.keys()):
    url = pdfs[id]
    try:
        wget.download(url, out=f"{pdf_folder}/{id}.pdf")
    except HTTPError as err:
        print("Http error")
        continue


