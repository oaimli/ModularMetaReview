import json
import jsonlines

if __name__ == "__main__":
    data_folder_peersum = "/home/miao4/punim0521/NeuralAbstractiveSummarization/ideas/peersum/crawling_data/data/"
    for conference in ["nips_2021", "nips_2022"]:
        with open(data_folder_peersum + f"{conference}.json") as f:
            papers = json.load(f)
        print(conference, len(papers))

        with jsonlines.open(f"../data/{conference}.jsonl", "w") as writer:
            writer.write_all(papers)
