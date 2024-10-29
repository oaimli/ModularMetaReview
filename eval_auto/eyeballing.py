import json

if __name__ == "__main__":
    with open("info.json") as f:
        info = json.load(f)

    dataset_names = ["space", "peermeta", "amasum_shoes"]

    for dataset_name in dataset_names:
        print(dataset_name)
        with open(f"{dataset_name}_summac_source_full.json") as f:
            all_scores_summac_source = json.load(f)
        with open(f"{dataset_name}_alignscore_full.json") as f:
            all_scores_alignscore_source = json.load(f)
        with open(f"llm_score/{dataset_name}_llm_scored_full.json") as f:
            all_scores_llm_score = json.load(f)
        with open(f"mini_score/{dataset_name}_llm_scored_full.json") as f:
            all_scores_mini_score = json.load(f)

        generations_info = info[dataset_name]
        outputs_combined = {}
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            with open(generation_file) as f:
                samples = json.load(f)
            scores_summac_source_reference = all_scores_summac_source["human_reference"]
            scores_summac_source_generation = all_scores_summac_source[generation_file]
            scores_alignscore_reference = all_scores_alignscore_source["human_reference"]
            scores_alignscore_generation = all_scores_alignscore_source[generation_file]
            scores_llm_score_reference = all_scores_llm_score["human_reference"]
            scores_llm_score_generation = all_scores_llm_score[generation_file]
            scores_mini_score_reference = all_scores_mini_score["human_reference"]
            scores_mini_score_generation = all_scores_mini_score[generation_file]

            print(len(scores_summac_source_reference))
            print(len(scores_summac_source_generation))
            print(len(scores_alignscore_reference))
            print(len(scores_alignscore_generation))
            print(len(scores_llm_score_reference))
            print(len(scores_llm_score_generation))
            print(len(scores_mini_score_reference))
            print(len(scores_mini_score_generation))

            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            i = 0
            for sample, score_generation_summac_source, score_reference_summac_source, score_generation_alignscore_source, score_reference_alignscore_source, score_generation_llm_score, score_reference_llm_score, score_generation_mini_score, score_reference_mini_score in zip(
                    samples,
                    scores_summac_source_generation,
                    scores_summac_source_reference,
                    scores_alignscore_generation,
                    scores_alignscore_reference,
                    scores_llm_score_generation,
                    scores_llm_score_reference,
                    scores_mini_score_generation,
                    scores_mini_score_reference):

                tmp = outputs_combined.get(str(i), {})
                if isinstance(sample[reference_key], str):
                    tmp["reference"] = {"text": sample[reference_key],
                                        "summac-source": score_reference_summac_source,
                                        "alignscore-source": score_reference_alignscore_source,
                                        "llm-score": score_reference_llm_score,
                                        "mini-score": score_reference_mini_score}
                else:
                    tmp["reference"] = {"text": sample[reference_key][0],
                                        "summac-source": score_reference_summac_source,
                                        "alignscore-source": score_reference_alignscore_source,
                                        "llm-score": score_reference_llm_score,
                                        "mini-score": score_reference_mini_score}

                existing_candidates = tmp.get("generations", [])
                existing_candidates.append({"model": generation_info["model_name"],
                                            "generation": sample[candidate_key],
                                            "summac-source": score_generation_summac_source,
                                            "alignscore-source": score_generation_alignscore_source,
                                            "llm-score": score_generation_llm_score,
                                            "mini-score": score_generation_mini_score})
                tmp["generations"] = existing_candidates
                tmp["source_documents"] = sample["source_documents"]
                outputs_combined[str(i)] = tmp
                i += 1

        output_file = f"{dataset_name}_generations_full.json"
        with open(output_file, "w") as f:
            json.dump(outputs_combined, f, indent=4)
