from metrics import summac_scores
import jsonlines
import numpy as np


if __name__ == "__main__":
    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
        One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
        The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
        Arcadia Planitia is in Mars' northern lowlands."""

    summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."

    scores_zs, scores_conv = summac_scores([document], [summary])
    print("scores zs", np.mean(scores_zs), "scores conv", np.mean(scores_conv))


    # generations = []
    # with jsonlines.open("/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/generations_zeroshot.jsonl") as reader:
    #     for line in reader:
    #         generations.append(line)
    #
    # from datasets import load_dataset
    # # load the dataset
    # all_data = load_dataset('json', data_files='/data/gpfs/projects/punim0521/MistralX/datasets/peersum.jsonl',
    #                         split='all')
    # test_data = all_data.filter(lambda s: s['label'] == 'test')
    # print("all test data", len(test_data))
    #
    # results = []
    # sources = []
    # for generation, sample in zip(generations, test_data):
    #     sample["generation"] = generation
    #     results.append(sample)
    #     sources.append("\n".join(sample["source"]))


    # output_file = "/data/gpfs/projects/punim0521/MistralX/results/mistral_7b_instruct_v02_peersum/predictions_zeroshot.jsonl"
    # documents = []
    # summaries = []
    # with jsonlines.open(output_file) as reader:
    #     for line in reader:
    #         documents.append("\n".join(line["source"]))
    #         summaries.append(line["summary"])
    #
    # scores_zs, scores_conv = summac_scores(documents, summaries)
    # print(np.mean(scores_zs), np.mean(scores_conv))


