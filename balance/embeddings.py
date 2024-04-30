import os
import random
import json
import time
import sys
import spacy
import openai
from tqdm import tqdm
import jsonlines


openai.api_key = "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"

human_written_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_zeroshot.jsonl"
zeroshot_file  = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_zeroshot.jsonl"
finetuned_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_finetuned.jsonl"
human_written_samples = []
with jsonlines.open(human_written_file) as reader:
    for line in reader:
        human_written_samples.append(line)
zeroshot_samples = []
with jsonlines.open(zeroshot_file) as reader:
    for line in reader:
        zeroshot_samples.append(line)
finetuned_samples = []
with jsonlines.open(finetuned_samples) as reader:
    for line in reader:
        finetuned_samples.append(line)

samples = []
for human_written_sample, zeroshot_sample, finetuned_sample in zip(human_written_samples, zeroshot_samples, finetuned_samples):
    if human_written_sample["summary"] == zeroshot_sample["summary"] == finetuned_sample["summary"]:
        human_written_summary = human_written_sample["summary"]
        del human_written_sample["generation"]
        del human_written_sample["summary"]
        human_written_sample["human_written"] = human_written_summary
        human_written_sample["mistral_7b_instruct_v02_zeroshot"] = zeroshot_sample["generation"]
        human_written_sample["mistral_7b_instruct_v02_finetuned"] = finetuned_sample["generation"]
        samples.append(human_written_sample)


nlp = spacy.load("en_core_web_sm")

for i, sample in tqdm(enumerate(samples), total=len(samples)):
    source_documents = sample["source"]

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

