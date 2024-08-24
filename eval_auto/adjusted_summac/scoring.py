import numpy as np
import json
from summac.model_summac import SummaCZS, SummaCConv
from openai import OpenAI
from typing import Dict


def summac_scores(sources, targets):
    model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda",
                            start_file="default", agg="mean")
    score_zs = model_zs.score(sources, targets)
    score_conv = model_conv.score(sources, targets)
    return score_zs["scores"], score_conv["scores"]


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    with open("../info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)

        generations_info = info[dataset_name]
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            source_key = "source_documents"

            # use the processed result with shared content from categorization
            shared_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + "_shared.json"
            with open(shared_file) as f:
                samples = json.load(f)

            candidates = []
            references = []
            source_texts = []
            references_shared = []
            candidates_shared = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                if isinstance(sample[reference_key], str):
                    references.append(sample[reference_key])
                else:
                    references.append(sample[reference_key][0])  # SPACE has multiple references
                source_texts.append("\n".join(sample[source_key]))

                references_shared.append(sample[reference_key + "_shared"])
                candidates_shared.append(sample[candidate_key + "_shared"])

            # compared with source texts
            scores_zs_source, scores_conv_source = summac_scores(source_texts, candidates)
            score_zs_source_avg = np.mean(scores_zs_source)
            score_conv_source_avg = np.mean(scores_conv_source)

            scores_zs_reference, scores_conv_reference = summac_scores(references_shared, candidates_shared)
            score_zs_reference_avg = np.mean(scores_zs_reference)
            score_conv_reference_avg = np.mean(scores_conv_reference)

            print("scores zs:", "source", score_zs_source_avg, "reference", score_zs_reference_avg, "summation",
                  score_zs_source_avg + score_zs_reference_avg)
            print("scores conv:", "source", score_conv_source_avg, "reference", score_conv_reference_avg, "summation",
                  score_conv_source_avg + score_conv_reference_avg)
