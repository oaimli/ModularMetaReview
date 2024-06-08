import jsonlines
import numpy as np
import sys

sys.path.append("../")
from utils.metrics import summac_scores


if __name__ == "__main__":
    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions. One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
        The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
        Arcadia Planitia is in Mars' northern lowlands."""

    summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."

    scores_zs, scores_conv = summac_scores([document], [summary])
    print("scores zs", np.mean(scores_zs), "scores conv", np.mean(scores_conv))

    # # evaluating human-written meta-reviews
    # output_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_zeroshot.jsonl"
    # documents = []
    # summaries = []
    # samples = []
    # with jsonlines.open(output_file) as reader:
    #     for line in reader:
    #         samples.append(line)
    #         documents.append("\n".join(line["source"]))
    #         summaries.append(line["summary"])
    #
    # scores_zs, scores_conv = summac_scores(documents, summaries)
    # print(np.mean(scores_zs), np.mean(scores_conv))
    #
    # results = []
    # for sample, zs, conv in zip(samples, scores_zs, scores_conv):
    #     sample["summac_zs"] = zs
    #     sample["summac_conv"] = conv
    #     results.append(sample)
    # with jsonlines.open("scores_peersum_human_written.jsonl", "w") as writer:
    #     writer.write_all(samples)
    # # -0.36093002860045564 0.49920685880405957

    # # evaluating human-written meta-reviews
    # output_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_zeroshot.jsonl"
    # documents = []
    # summaries = []
    # samples = []
    # with jsonlines.open(output_file) as reader:
    #     for line in reader:
    #         samples.append(line)
    #         documents.append("\n".join(line["source"]))
    #         summaries.append(line["generation"])
    #
    # scores_zs, scores_conv = summac_scores(documents, summaries)
    # print(np.mean(scores_zs), np.mean(scores_conv))
    #
    # results = []
    # for sample, zs, conv in zip(samples, scores_zs, scores_conv):
    #     sample["summac_zs"] = zs
    #     sample["summac_conv"] = conv
    #     results.append(sample)
    # with jsonlines.open("scores_peersum_mistral_7b_instruct_v02_zeroshot.jsonl", "w") as writer:
    #     writer.write_all(samples)
    # # -0.2107997606339326 0.48214757963272425

    # evaluating human-written meta-reviews
    output_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_finetuned.jsonl"
    documents = []
    summaries = []
    samples = []
    with jsonlines.open(output_file) as reader:
        for line in reader:
            samples.append(line)
            documents.append("\n".join(line["source"]))
            summaries.append(line["generation"])

    scores_zs, scores_conv = summac_scores(documents, summaries)
    print(np.mean(scores_zs), np.mean(scores_conv))

    results = []
    for sample, zs, conv in zip(samples, scores_zs, scores_conv):
        sample["summac_zs"] = zs
        sample["summac_conv"] = conv
        results.append(sample)
    with jsonlines.open("scores_peersum_mistral_7b_instruct_v02_finetuned.jsonl", "w") as writer:
        writer.write_all(samples)
    # -0.19557023163739445 0.6354053034140819


