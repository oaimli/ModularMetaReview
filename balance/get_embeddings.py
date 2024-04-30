import os
import random
import json
import time
import sys
import spacy
import openai
from tqdm import tqdm
import jsonlines

dataset_name = "pseudo_shepherd_1455890"

all_samples = []
with jsonlines.open("../../dataset/%s.json" % dataset_name) as reader:
    for line in reader:
        all_samples.append(line)
random.seed(42)
all_samples = random.sample(all_samples, 50)
print("all data", len(all_samples))

openai.api_key = "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"
nlp = spacy.load("en_core_web_sm")

save_path = "../../embeddings/%s_openai" % dataset_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i, sample in tqdm(enumerate(all_samples), total=len(all_samples)):
    source_document = sample["source_documents"][0]

    sentences_source = []
    summary_words = source_document.split()
    for sent in nlp(" ".join(summary_words)).sents:
        sentences_source.append(sent.text)

    sentences_embeddings = {}
    for sentence in sentences_source:
        embedding_error = True
        sentence_embedding = []
        while len(sentence_embedding) == 0:
            try:
                sentence_embedding = \
                    openai.Embedding.create(input=sentence, model="text-embedding-ada-002")["data"][0][
                        "embedding"]
            except:
                error = sys.exc_info()[0]
                print("API error:", error)
                sentence_embedding = []
                time.sleep(1)
        sentences_embeddings[sentence] = sentence_embedding

    with open(os.path.join(save_path, "%d.json" % i), "w") as f:
        json.dump(sentences_embeddings, f)

print("done")
