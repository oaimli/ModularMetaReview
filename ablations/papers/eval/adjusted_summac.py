import numpy as np
import json
from summac.model_summac import SummaCZS, SummaCConv


def summac_scores(sources, targets):
    model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda",
                            start_file="default", agg="mean")
    score_zs = model_zs.score(sources, targets)
    score_conv = model_conv.score(sources, targets)
    return score_zs["scores"], score_conv["scores"]


if __name__ == "__main__":
    model_name = "llama31_70b"

    with open("info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta"]
    for dataset_name in dataset_names:
        print(dataset_name)
        facets = []
        if dataset_name == "peermeta":
            facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
        if dataset_name == "space":
            facets = ["Building", "Cleanliness", "Food", "Location", "Rooms", "Service"]
        if dataset_name == "amasum_shoes":
            facets = ["Breathability", "Durability", "Weight", "Cushioning", "Stability", "Flexibility", "Traction",
                      "Sizefit", "Comfort", "Misc"]

        generations_info = info[dataset_name]

        # human reference
        generation_file = generations_info[0]["generation_file"]
        print("human reference")
        reference_key = generations_info[0]["reference_key"]
        source_key = "source_documents"
        with open(generation_file) as f:
            samples = json.load(f)
        references = []
        source_texts = []
        for sample_key, sample in samples.items():
            if isinstance(sample[reference_key], str):
                references.append(sample[reference_key])
            else:
                references.append(sample[reference_key][0])  # SPACE has multiple references
            source_documents = []
            for review in sample["reviews"]:
                source_documents.append(review["comment"])
            source_texts.append("\n".join(source_documents))
        # compared with source texts
        scores_zs_source, scores_conv_source = summac_scores(source_texts, references)
        score_zs_source_avg = np.mean(scores_zs_source)
        score_conv_source_avg = np.mean(scores_conv_source)
        print("scores zs:", "source", score_zs_source_avg)
        print("scores conv:", "source", score_conv_source_avg)

        # model generations
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            source_key = "reviews"

            # use the processed result with shared content from categorization
            categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
            with open("categorization/" + categorization_file) as f:
                samples = json.load(f)

            candidates = []
            references = []
            source_texts = []
            references_shared = []
            candidates_shared = []
            for sample_index, sample in samples.items():
                candidates.append(sample[candidate_key])
                if isinstance(sample[reference_key], str):
                    references.append(sample[reference_key])
                else:
                    references.append(sample[reference_key][0])  # SPACE has multiple references
                source_documents = []
                for review in sample["reviews"]:
                    source_documents.append(review["comment"])
                source_texts.append("\n".join(source_documents))

                categorization_reference = sample["categorization_reference"]
                categorization_candidate = sample["categorization_candidate"]
                reference_shared = []
                candidate_shared = []
                for facet in facets:
                    if len(categorization_reference[facet]) > 0 and len(categorization_candidate[facet]) > 0:
                        reference_shared.extend(categorization_reference[facet])
                        candidate_shared.extend(categorization_candidate[facet])

                references_shared.append(" ".join(reference_shared))
                candidates_shared.append(" ".join(candidate_shared))

            # compared with source texts
            scores_zs_source, scores_conv_source = summac_scores(source_texts, candidates)
            score_zs_source_avg = np.mean(scores_zs_source)
            score_conv_source_avg = np.mean(scores_conv_source)

            # compared with reference on shared aspects
            scores_zs_reference, scores_conv_reference = summac_scores(references_shared, candidates_shared)
            score_zs_reference_avg = np.mean(scores_zs_reference)
            score_conv_reference_avg = np.mean(scores_conv_reference)

            print("scores zs:", "source", score_zs_source_avg, "reference", score_zs_reference_avg, "summation",
                  score_zs_source_avg + score_zs_reference_avg)
            print("scores conv:", "source", score_conv_source_avg, "reference", score_conv_reference_avg, "summation",
                  score_conv_source_avg + score_conv_reference_avg)
