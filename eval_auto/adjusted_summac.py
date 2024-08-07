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
    reference_key = ""
    candidate_key = ""
    source_key = "source_documents"
    generation_file = ""

    with open(generation_file) as f:
        samples = json.load(f)

    candidates = []
    references = []
    source_texts = []
    for sample in samples:
        candidates.append(sample[candidate_key])
        if isinstance(sample[reference_key], str):
            references.append(sample[reference_key])
        else:
            references.append(sample[reference_key][0])  # SPACE has multiple references
        source_texts.append(sample[source_key])

    # compared with source texts
    scores_zs_source, scores_conv_source = summac_scores(source_texts, candidates)
    score_zs_source_avg = np.mean(scores_zs_source)
    score_conv_source_avg = np.mean(scores_conv_source)

    # compared with references
    scores_zs_reference, scores_conv_reference = summac_scores(references, candidates)
    score_zs_reference_avg = np.mean(scores_zs_reference)
    score_conv_reference_avg = np.mean(scores_conv_reference)

    print("scores zs:", "source", score_zs_source_avg, "reference", score_zs_reference_avg, "summation",
          score_zs_source_avg + score_zs_reference_avg)
    print("scores conv:", "source", score_conv_source_avg, "reference", score_conv_reference_avg, "summation",
          score_conv_source_avg + score_conv_reference_avg)
