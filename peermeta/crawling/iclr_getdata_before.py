import json
import random
import jsonlines

if __name__ == "__main__":
    data_folder_peersum = "/home/miao4/punim0521/NeuralAbstractiveSummarization/ideas/peersum/crawling_data/data/"
    conference = "iclr_2018"
    with open(data_folder_peersum + f"{conference}.json") as f:
        papers = json.load(f)
    print(len(papers))

    # papers_new = []
    # for paper in papers:
    #     if paper.get("paper_id", "") == "":
    #         print("####")
    #     if paper.get("paper_title", "") == "":
    #         print("####")
    #     if paper.get("paper_abstract", "") == "":
    #         print("####")
    #     if paper.get("paper_acceptance", "") == "":
    #         print("####")
    #     if paper.get("meta_review", "") == "":
    #         print("####")
    #     if len(paper.get("reviews", [])) == 0:
    #         print("####")
    #
    # with jsonlines.open(f"../data/{conference}.jsonl", "w") as writer:
    #     writer.write_all(papers_new)
