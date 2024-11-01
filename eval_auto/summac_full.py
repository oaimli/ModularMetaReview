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
    with open("all.json") as f:
        info = json.load(f)

    # dataset_names = ["space", "peermeta", "amasum_shoes"]
    dataset_names = ["amasum_shoes"]
    for dataset_name in dataset_names:
        scores_source = {}
        scores_reference = {}
        scores_source_aspect = {}
        scores_reference_aspect = {}

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

        # load categorization data of reviews
        generation_file_tmp = ""
        for generation_info in generations_info:
            if generation_info["model_name"] == "modular_llama3":
                generation_file_tmp = generation_info["generation_file"]
        categorization_file = "_".join(generation_file_tmp.split("/")[1:]).split(".")[0] + ".json"
        with open("categorization/" + categorization_file) as f:
            samples = json.load(f)
        review_categorizations = []
        for sample in samples:
            review_categorizations.append(sample["review_categorization"])

        # human reference
        generation_file = generations_info[0]["generation_file"]
        categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
        with open("categorization/" + categorization_file) as f:
            samples = json.load(f)
        print("human reference")
        reference_key = generations_info[0]["reference_key"]
        source_key = "source_documents"

        references = []
        source_texts = []
        references_shared = []
        source_texts_shared = []
        for sample, categorization_reviews in zip(samples, review_categorizations):
            if isinstance(sample[reference_key], str):
                reference = sample[reference_key]
            else:
                reference = sample[reference_key][0]  # SPACE has multiple references
            references.append(reference)
            source_texts.append("\n".join(sample[source_key]))
            categorization_reference = sample["categorization_reference"]
            reference_shared = []
            source_text_shared = []
            for facet in facets:
                if len(categorization_reference[facet]) > 0:
                    tmp = []
                    for categorization_review in categorization_reviews:
                        tmp.extend(categorization_review[facet])
                    if len(tmp) > 0:
                        reference_shared.extend(categorization_reference[facet])
                        source_text_shared.extend(tmp)
            if len(reference_shared) > 0 and len(source_text_shared) > 0:
                references_shared.append(" ".join(reference_shared))
                source_texts_shared.append(" ".join(source_text_shared))

        # reference compared with source texts
        scores_zs_source, scores_conv_source = summac_scores(source_texts, references)
        scores_source["human_reference"] = scores_conv_source
        score_zs_source_avg = np.mean(scores_zs_source)
        score_conv_source_avg = np.mean(scores_conv_source)
        print("source", "scores zs:", score_zs_source_avg, "scores conv:", score_conv_source_avg)

        # reference compared with source texts on shared aspects
        scores_zs_source, scores_conv_source = summac_scores(source_texts_shared, references_shared)
        scores_source_aspect["human_reference"] = scores_conv_source
        score_zs_source_avg = np.mean(scores_zs_source)
        score_conv_source_avg = np.mean(scores_conv_source)
        print(len(references_shared), source_texts_shared)
        print("source aspect", "scores zs:", score_zs_source_avg, "scores conv:", score_conv_source_avg)

        # model generations
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            source_key = "source_documents"

            # use the processed result with shared content from categorization
            categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
            with open("categorization/" + categorization_file) as f:
                samples = json.load(f)

            candidates = []
            references = []
            source_texts = []
            references_shared = []
            candidates_shared_reference = []
            candidates_shared_source = []
            source_texts_shared = []

            for sample, categorization_reviews in zip(samples, review_categorizations):
                candidates.append(sample[candidate_key])
                if isinstance(sample[reference_key], str):
                    references.append(sample[reference_key])
                else:
                    references.append(sample[reference_key][0])  # SPACE has multiple references
                source_texts.append("\n".join(sample[source_key]))

                categorization_reference = sample["categorization_reference"]
                categorization_candidate = sample["categorization_candidate"]
                reference_shared = []
                candidate_shared_reference = []
                for facet in facets:
                    if len(categorization_reference[facet]) > 0 and len(categorization_candidate[facet]) > 0:
                        reference_shared.extend(categorization_reference[facet])
                        candidate_shared_reference.extend(categorization_candidate[facet])

                if len(reference_shared) > 0 and len(candidate_shared_reference) > 0:
                    references_shared.append(" ".join(reference_shared))
                    candidates_shared_reference.append(" ".join(candidate_shared_reference))

                candidate_shared_source = []
                source_text_shared = []
                for facet in facets:
                    if len(categorization_candidate[facet]) > 0:
                        tmp = []
                        for categorization_review in categorization_reviews:
                            tmp.extend(categorization_review[facet])
                        if len(tmp) > 0:
                            candidate_shared_source.extend(categorization_candidate[facet])
                            source_text_shared.extend(tmp)
                if len(candidate_shared_source) > 0 and len(source_text_shared):
                    candidates_shared_source.append(" ".join(candidate_shared_source))
                    source_texts_shared.append(" ".join(source_text_shared))

            # compared with source texts on shared aspects
            scores_zs_source_aspect, scores_conv_source_aspect = summac_scores(source_texts_shared, candidates_shared_source)
            scores_source_aspect[generation_file] = scores_conv_source_aspect
            score_zs_source_aspect_avg = np.mean(scores_zs_source)
            score_conv_source_aspect_avg = np.mean(scores_conv_source)
            print(len(source_texts_shared), len(candidates_shared_source))
            print("source aspect", "scores zs:", score_zs_source_avg, "scores conv:", score_conv_source_avg)


            # compared with source texts
            scores_zs_source, scores_conv_source = summac_scores(source_texts, candidates)
            scores_source[generation_file] = scores_conv_source
            score_zs_source_avg = np.mean(scores_zs_source)
            score_conv_source_avg = np.mean(scores_conv_source)
            print("source", "scores zs:", score_zs_source_avg, "scores conv:", score_conv_source_avg)

            # compared with reference on shared aspects
            scores_zs_reference_aspect, scores_conv_reference_aspect = summac_scores(references_shared, candidates_shared_reference)
            scores_reference_aspect[generation_file] = scores_conv_source
            score_zs_reference_aspect_avg = np.mean(scores_zs_reference_aspect)
            score_conv_reference_aspect_avg = np.mean(scores_conv_reference_aspect)
            print(len(references_shared), len(candidates_shared_reference))
            print("reference aspect", "scores zs:", score_zs_reference_aspect_avg, "scores conv:", score_conv_reference_aspect_avg)

            # compared with the whole reference
            scores_zs_reference, scores_conv_reference = summac_scores(references, candidates)
            scores_reference[generation_file] = scores_conv_reference
            score_zs_reference_avg = np.mean(scores_zs_reference)
            score_conv_reference_avg = np.mean(scores_conv_reference)
            print("reference", "scores zs:", score_zs_reference_avg, "scores conv:", score_conv_reference_avg)

            # compare reference with itself
            scores_zs_reference_self, scores_conv_reference_self = summac_scores(references, references)
            score_zs_reference_self_avg = np.mean(scores_zs_reference_self)
            score_conv_reference_self_avg = np.mean(scores_conv_reference_self)
            print("reference self", "scores zs:", score_zs_reference_self_avg, "scores conv:", score_conv_reference_self_avg)

        with open(f"{dataset_name}_summac_reference_full.json", "w") as f:
            json.dump(scores_reference, f, indent=4)
        with open(f"{dataset_name}_summac_reference_aspect_full.json", "w") as f:
            json.dump(scores_reference_aspect, f, indent=4)
        with open(f"{dataset_name}_summac_source_full.json", "w") as f:
            json.dump(scores_source, f, indent=4)
        with open(f"{dataset_name}_summac_source_aspect_full.json", "w") as f:
            json.dump(scores_source_aspect, f, indent=4)
