import jsonlines
import numpy as np

dataset_files = [
    "amasum_shoes_test.jsonl",
    "amasum_shoes_train.jsonl",
    "amasum_shoes_valid.jsonl",
    "peermeta_dev.jsonl",
    "peermeta_test.jsonl",
    "peermeta_train.jsonl",
    "space_dev.jsonl",
    "space_test.jsonl"
    ]

for dataset_file in dataset_files:
    print(dataset_file)
    samples = []
    with jsonlines.open(dataset_file) as reader:
        for line in reader:
            samples.append(line)

    counts_reviews = []
    lengths_source = []
    lengths_meta = []
    for sample in samples:
        counts_reviews.append(len(sample["source_documents"]))
        lengths_source.append(len(" ".join(sample["source_documents"]).split()))

        meta_reiview = ""
        if sample.get("meta_review", "") == "":
            meta_review = sample["gold_summaries_general"][0]
        else:
            meta_review = sample["meta_review"]
        lengths_meta.append(len(meta_review.split()))

    print(np.mean(counts_reviews), np.mean(lengths_source), np.mean(lengths_meta))
