import numpy as np
import json
from alignscore import AlignScore


if __name__ == "__main__":
    scorer = AlignScore(model='roberta-large', batch_size=8, device='cuda:0',
                        ckpt_path='../../../tmp/AlignScore/AlignScore-large.ckpt',
                        evaluation_mode='nli_sp')

    with open("info.json") as f:
        info = json.load(f)

    # dataset_names = ["space", "peermeta", "amasum_shoes"]
    dataset_names = ["space"]
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
        scores_align = scorer.score(contexts=source_texts, claims=references)
        scores_source["human_reference"] = scores_align
        score_avg = np.mean(scores_align)
        print("source", score_avg)

        # reference compared with source texts on shared aspects
        scores_align = scorer.score(contexts=source_texts_shared, claims=references_shared)
        scores_source_aspect["human_reference"] = scores_align
        score_avg = np.mean(scores_align)
        print("source aspect", score_avg)

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
            scores_align = scorer.score(contexts=source_texts_shared, claims=candidates_shared_source)
            scores_source_aspect[generation_file] = scores_align
            score_avg = np.mean(scores_align)
            print("source aspect", score_avg, len(source_texts_shared), len(candidates_shared_source))

            # compared with source texts
            scores_align = scorer.score(contexts=source_texts, claims=candidates)
            scores_source[generation_file] = scores_align
            score_avg = np.mean(scores_align)
            print("source", score_avg)

            # compared with reference on shared aspects
            scores_align = scorer.score(contexts=references_shared, claims=candidates_shared_reference)
            scores_reference_aspect[generation_file] = scores_align
            score_avg = np.mean(scores_align)
            print("reference aspect", score_avg, len(references_shared), len(candidates_shared_reference))

            # compared with the whole reference
            scores_align = scorer.score(contexts=references, claims=candidates)
            scores_reference[generation_file] = scores_align
            score_avg = np.mean(scores_align)
            print("reference", score_avg)

            # compare reference with itself
            scores_align = scorer.score(contexts=references, claims=references)
            score_avg = np.mean(scores_align)
            print("reference itself", score_avg)

        with open(f"{dataset_name}_alignscore_reference_full.json", "w") as f:
            json.dump(scores_reference, f, indent=4)
        print("scores_reference")
        for generation_name, score in scores_reference.items():
            print(generation_name)
            print(np.mean(score))
        with open(f"{dataset_name}_alignscore_reference_aspect_full.json", "w") as f:
            json.dump(scores_reference_aspect, f, indent=4)
        print("scores_reference_aspect")
        for generation_name, score in scores_reference_aspect.items():
            print(generation_name)
            print(np.mean(score))
        with open(f"{dataset_name}_alignscore_source_full.json", "w") as f:
            json.dump(scores_source, f, indent=4)
        print("scores_source")
        for generation_name, score in scores_source.items():
            print(generation_name)
            print(np.mean(score))
        with open(f"{dataset_name}_alignscore_source_aspect_full.json", "w") as f:
            json.dump(scores_source_aspect, f, indent=4)
        print("scores_source_aspect")
        for generation_name, score in scores_source_aspect.items():
            print(generation_name)
            print(np.mean(score))
