import jsonlines
from numpy import dot
from numpy.linalg import norm
import numpy as np


def cosine_sim(embedding_a, embedding_b):
    return dot(embedding_a, embedding_b) / (norm(embedding_a) * norm(embedding_b))


if __name__ == "__main__":
    samples = []
    with jsonlines.open("peersum_embeddings.jsonl") as reader:
        for line in reader:
            samples.append(line)

    # document level
    win_docs_human_written = []
    win_docs_zeroshot = []
    win_docs_finetuned = []
    # sentence level
    dis_zeroshot_to_human = []
    dis_finetuned_to_human = []
    dis_zeroshot_to_finetuned = []
    for sample in samples:
        source_embeddings_all = []  # m*emb
        source_counts = []
        for embeddings in sample["source_embeddings"]:
            source_embeddings_all.extend(embeddings)
            source_counts.append(len(embeddings))
        human_written_embeddings = sample["human_written_embeddings"]  # h*emb
        zeroshot_embeddings = sample["mistral_7b_instruct_v02_zeroshot_sentences"]  # z*emb
        finetuned_embeddings = sample["mistral_7b_instruct_v02_finetuned_embeddings"]  # f*emb

        human_written_balance = []  # m
        zeroshot_balance = []  # m
        finetuned_balance = []  # m
        for source_embedding in source_embeddings_all:
            human_written_sims = []
            for human_written_embedding in human_written_embeddings:
                human_written_sims.append(cosine_sim(source_embedding, human_written_embedding))
            human_written_balance.append(np.mean(human_written_sims))

            zeroshot_sims = []
            for zeroshot_embedding in zeroshot_embeddings:
                zeroshot_sims.append(cosine_sim(source_embedding, zeroshot_embedding))
            zeroshot_balance.append(np.mean(zeroshot_sims))

            finetuned_sims = []
            for finetuned_embedding in finetuned_embeddings:
                finetuned_sims.append(cosine_sim(source_embedding, finetuned_embedding))
            finetuned_balance.append(np.mean(finetuned_sims))

        dis_zeroshot_to_human.append(cosine_sim(zeroshot_balance, human_written_balance))
        dis_finetuned_to_human.append(cosine_sim(finetuned_balance, human_written_balance))
        dis_zeroshot_to_finetuned.append(cosine_sim(zeroshot_balance, finetuned_balance))

        human_written_balance_doc = []
        zeroshot_balance_doc = []
        finetuned_balance_doc = []
        start = 0
        for count in source_counts:
            human_written_balance_doc.append(np.mean(human_written_balance[start: start + count]))
            zeroshot_balance_doc.append(np.mean(zeroshot_balance[start: start + count]))
            finetuned_balance_doc.append(np.mean(finetuned_balance[start: start + count]))
            start = start + count
        win_docs_human_written.append(human_written_balance_doc.index(max(human_written_balance_doc)))
        win_docs_zeroshot.append(zeroshot_balance_doc.index(max(zeroshot_balance_doc)))
        win_docs_finetuned.append(finetuned_balance_doc.index(max(finetuned_balance_doc)))

    print("Sharing same win doc, zeroshot and human")
    share = 0
    for win_doc_zeroshot, win_doc_human_written in zip(win_docs_zeroshot, win_docs_human_written):
        if win_doc_zeroshot == win_doc_human_written:
            share += 1
    print(share/len(win_docs_human_written))

    print("Sharing same win doc, finetuned and human")
    share = 0
    for win_doc_finetuned, win_doc_human_written in zip(win_docs_finetuned, win_docs_human_written):
        if win_doc_finetuned == win_doc_human_written:
            share += 1
    print(share / len(win_docs_human_written))

    print("Sharing same win doc, finetuned and zeroshot")
    share = 0
    for win_doc_finetuned, win_doc_zeroshot in zip(win_docs_finetuned, win_docs_zeroshot):
        if win_doc_finetuned == win_doc_zeroshot:
            share += 1
    print(share / len(win_docs_zeroshot))

    print("Average distance of balances, zeroshot and human", np.mean(dis_zeroshot_to_human))
    print("Average distance of balances, finetuned and human", np.mean(dis_finetuned_to_human))
    print("Average distance of balances, zeroshot and finetuned", np.mean(dis_zeroshot_to_finetuned))