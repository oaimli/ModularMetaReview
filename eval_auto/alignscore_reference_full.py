import numpy as np
import json
from alignscore import AlignScore


if __name__ == "__main__":
    scorer = AlignScore(model='roberta-large', batch_size=64, device='cuda:0', ckpt_path='../tmp/AlignScore/AlignScore-large.ckpt',
                        evaluation_mode='nli_sp')

    with open("info.json") as f:
        info = json.load(f)

    dataset_names = ["space", "peermeta", "amasum_shoes"]
    # dataset_names = ["amasum_shoes"]
    for dataset_name in dataset_names:
        reference_score = {} # alignscore based on the whole references
        reference_score_aspect = {} # alignscore based on shared aspects
        print(dataset_name)
        if dataset_name == "peermeta":
            facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
        elif dataset_name == "space":
            facets = ["Building", "Cleanliness", "Food", "Location", "Rooms", "Service"]
        elif dataset_name == "amasum_shoes":
            facets = ["Breathability", "Durability", "Weight", "Cushioning", "Stability", "Flexibility", "Traction",
                      "Sizefit", "Comfort", "Misc"]
        else:
            facets = []
        print("facets", facets)

        generations_info = info[dataset_name]
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            # use the processed result with shared content from categorization
            categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
            print(categorization_file)
            with open("categorization/" + categorization_file) as f:
                samples = json.load(f)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]

            candidates = []
            references = []
            candidates_shared = []
            references_shared = []
            for sample in samples:
                # print(sample.keys())
                candidates.append(sample[candidate_key])
                references.append(sample[reference_key])

                categorization_reference = sample["categorization_reference"]
                categorization_candidate = sample["categorization_candidate"]
                reference_shared = []
                candidate_shared = []
                for facet in facets:
                    if len(categorization_reference[facet]) > 0 and len(categorization_candidate[facet]) > 0:
                        reference_shared.extend(categorization_reference[facet])
                        candidate_shared.extend(categorization_candidate[facet])
                if len(reference_shared) > 0 and len(candidate_shared) > 0:
                    references_shared.append(" ".join(reference_shared))
                    candidates_shared.append(" ".join(candidate_shared))

            scores_align = scorer.score(contexts=references, claims=candidates)
            reference_score[generation_file] = scores_align
            print("scores alignscore reference", np.mean(scores_align))

            if len(references_shared) > 0 and len(candidates_shared) > 0:
                print(len(candidates_shared), len(samples))
                scores_align = scorer.score(contexts=references_shared, claims=candidates_shared)
                reference_score_aspect[generation_file] = scores_align
                print("scores alignscore reference based on shared aspects", np.mean(scores_align))
            else:
                print(len(candidates_shared), len(samples))
                print("There are no samples applicable.")

        # with open(f"{dataset_name}_alignscore_reference_full.json", "w") as f:
        #     json.dump(reference_source, f, indent=4)

        for generation_name, score_source in reference_score.items():
            print(generation_name)
            print(np.mean(score_source))
