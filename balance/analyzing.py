import jsonlines


samples = []
with jsonlines.open("peersum_embeddings.jsonl") as reader:
    for line in reader:
        samples.append(line)

