import os
import spacy
from tqdm import tqdm
import jsonlines

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    human_written_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_zeroshot.jsonl"
    mistral_7b_instruct_v02_zeroshot_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_zeroshot.jsonl"
    mistral_7b_instruct_v02_finetuned_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_finetuned.jsonl"
    human_written_samples = []
    with jsonlines.open(human_written_file) as reader:
        for line in reader:
            human_written_samples.append(line)
    mistral_7b_instruct_v02_zeroshot_samples = []
    with jsonlines.open(mistral_7b_instruct_v02_zeroshot_file) as reader:
        for line in reader:
            mistral_7b_instruct_v02_zeroshot_samples.append(line)
    mistral_7b_instruct_v02_finetuned_samples = []
    with jsonlines.open(mistral_7b_instruct_v02_finetuned_file) as reader:
        for line in reader:
            mistral_7b_instruct_v02_finetuned_samples.append(line)

    samples = []
    for human_written_sample, mistral_7b_instruct_v02_zeroshot_sample, mistral_7b_instruct_v02_finetuned_sample in zip(
            human_written_samples, mistral_7b_instruct_v02_zeroshot_samples, mistral_7b_instruct_v02_finetuned_samples):
        if human_written_sample["summary"] == mistral_7b_instruct_v02_zeroshot_sample["summary"] == \
                mistral_7b_instruct_v02_finetuned_sample["summary"]:
            human_written_summary = human_written_sample["summary"]
            del human_written_sample["generation"]
            del human_written_sample["summary"]
            human_written_sample["human_written"] = human_written_summary
            human_written_sample["mistral_7b_instruct_v02_zeroshot"] = mistral_7b_instruct_v02_zeroshot_sample[
                "generation"]
            human_written_sample["mistral_7b_instruct_v02_finetuned"] = mistral_7b_instruct_v02_finetuned_sample[
                "generation"]
            samples.append(human_written_sample)

    results = []
    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        source_documents = sample["source"]
        human_written = sample["human_written"]
        mistral_7b_instruct_v02_zeroshot = sample["mistral_7b_instruct_v02_zeroshot"]
        mistral_7b_instruct_v02_finetuned = sample["mistral_7b_instruct_v02_finetuned"]

        source_sentences = []
        for source_document in source_documents:
            sentences = []
            for sent in nlp(source_document).sents:
                sentences.append(sent.text)
            source_sentences.append(sentences)
        sample["source_sentences"] = source_sentences

        human_written_sentences = []
        for sent in nlp(human_written).sents:
            human_written_sentences.append(sent.text)
        sample["human_written_sentences"] = human_written_sentences

        mistral_7b_instruct_v02_zeroshot_sentences = []
        for sent in nlp(mistral_7b_instruct_v02_zeroshot).sents:
            mistral_7b_instruct_v02_zeroshot_sentences.append(sent.text)
        sample["mistral_7b_instruct_v02_zeroshot_sentences"] = mistral_7b_instruct_v02_zeroshot_sentences

        mistral_7b_instruct_v02_finetuned_sentences = []
        for sent in nlp(mistral_7b_instruct_v02_finetuned).sents:
            mistral_7b_instruct_v02_finetuned_sentences.append(sent.text)
        sample["mistral_7b_instruct_v02_finetuned_sentences"] = mistral_7b_instruct_v02_finetuned_sentences

        results.append(sample)

    with jsonlines.open(os.path.join("peersum_sentences.jsonl"), "w") as writer:
        writer.write_all(results)
