import os
import sys
import spacy
import openai
from tqdm import tqdm
import jsonlines
import random
from openai import OpenAI

sys.path.append("../")
from utils.get_embeddings import get_embeddings

nlp = spacy.load("en_core_web_sm")
openai.api_key = "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"
client = OpenAI(api_key="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")

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
with jsonlines.open(finetuned_file) as reader:
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

samples = random.sample(samples, 512)

results = []
for i, sample in tqdm(enumerate(samples), total=len(samples)):
    source_documents = sample["source"]
    human_written = sample["human_written"]
    zeroshot = sample["mistral_7b_instruct_v02_zeroshot"]
    finetuned = sample["mistral_7b_instruct_v02_finetuned"]

    source_sentences = []
    source_embeddings = []
    for source_document in source_documents:
        sentences = []
        for sent in nlp(source_document).sents:
            sentences.append(sent.text)
        source_sentences.append(sentences)
        source_embeddings.append(get_embeddings(sentences, client))
        assert len(source_sentences) == len(source_embeddings)
    sample["source_sentences"] = source_sentences
    sample["source_embeddings"] = source_embeddings

    human_written_sentences = []
    for sent in nlp(human_written).sents:
        human_written_sentences.append(sent.text)
    human_written_embeddings = get_embeddings(human_written_sentences, client)
    assert len(human_written_sentences) == len(human_written_embeddings)
    sample["human_written_sentences"] = human_written_sentences
    sample["human_written_embeddings"] = human_written_embeddings

    zeroshot_sentences = []
    for sent in nlp(zeroshot).sents:
        zeroshot_sentences.append(sent.text)
    zeroshot_embeddings = get_embeddings(zeroshot_sentences, client)
    assert len(zeroshot_embeddings) == len(zeroshot_sentences)
    sample["mistral_7b_instruct_v02_zeroshot_sentences"] = zeroshot_sentences
    sample["mistral_7b_instruct_v02_zeroshot_embeddings"] = zeroshot_embeddings

    finetuned_sentences = []
    for sent in nlp(finetuned).sents:
        finetuned_sentences.append(sent.text)
    finetuned_embeddings = get_embeddings(finetuned_sentences, client)
    assert len(finetuned_embeddings) == len(finetuned_sentences)
    sample["mistral_7b_instruct_v02_finetuned_sentences"] = finetuned_sentences
    sample["mistral_7b_instruct_v02_finetuned_embeddings"] = finetuned_embeddings

    results.append(sample)

with jsonlines.open(os.path.join("peersum_embeddings.jsonl"), "w") as writer:
    writer.write_all(results)

