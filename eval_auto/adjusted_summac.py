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


def llama3_prompting(input_text: str, facet: str, mode: str = "meta"):
    print("#######")
    prompt_format = open(f"{prompt_folder}/prompt_{mode.lower()}_{facet.lower()}.txt").read()
    prompt_content = prompt_format.replace("{{input_document}}", input_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "You are requested to do some extraction work. You must output the answer following the format of the example output, without any other useless content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=10
                )

            all_candidates = []
            all_candidates_len = []
            tmp = []
            for choice in output_dict.choices:
                output_content = choice.message.content
                if "no related fragments" not in output_content.lower():
                    content_len = len(output_content.split())
                    all_candidates_len.append(content_len)
                    tmp.append(content_len)
                    all_candidates.append(output_content.split("\n"))
            if len(all_candidates) < 5:
                outputs = []
            else:
                tmp.sort()
                outputs = all_candidates[all_candidates_len.index(tmp[int(len(tmp) / 2)])]
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)

    return outputs


def categorizing_meta_review(meta_review: str) -> Dict:
    """
    Args:
        meta_review: the meta-review of a sample
        model_name: the name of a model
    Returns:
        result: a dictionary of extracted fragments for different review facets
    """
    result = {}
    for facet in facets:
        result[facet] = llama3_prompting(meta_review, facet, "meta")

    return result


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    with open("info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)

        facets = []
        if dataset_name == "peermeta":
            facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
            prompt_folder = "../optimization/organization/prompts_scientific"
        if dataset_name == "space":
            facets = ["Building", "Cleanliness", "Food", "Location", "Rooms", "Service"]
            prompt_folder = "../modular_llama3/hotels/prompts_organization"
        if dataset_name == "amasum_shoes":
            facets = ["Breathability", "Durability", "Weight", "Cushioning", "Stability", "Flexibility", "Traction", "Sizefit", "Comfort", "Misc"]
            prompt_folder = "../modular_llama3/shoes/prompts_organization"

        generations_info = info[dataset_name]
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            source_key = "source_documents"
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
                source_texts.append("\n".join(sample[source_key]))

            # compared with source texts
            scores_zs_source, scores_conv_source = summac_scores(source_texts, candidates)
            score_zs_source_avg = np.mean(scores_zs_source)
            score_conv_source_avg = np.mean(scores_conv_source)

            # compared with references on only shared aspects
            references_shared = []
            candidates_shared = []
            for reference, candidate in zip(references, candidates):
                categorization_reference = categorizing_meta_review(reference)
                categorization_candidate = categorizing_meta_review(candidate)
                reference_shared = []
                candidate_shared = []
                for facet in facets:
                    if len(categorization_reference[facet]) > 0 and len(categorization_candidate[facet]) > 0:
                        reference_shared.extend(categorization_reference[facet])
                        candidate_shared.extend(categorization_candidate[facet])
                reference_shared.append(" ".join(reference_shared))
                candidates_shared.append(" ".join(candidate_shared))

            scores_zs_reference, scores_conv_reference = summac_scores(references_shared, candidates_shared)
            score_zs_reference_avg = np.mean(scores_zs_reference)
            score_conv_reference_avg = np.mean(scores_conv_reference)

            print("scores zs:", "source", score_zs_source_avg, "reference", score_zs_reference_avg, "summation",
                  score_zs_source_avg + score_zs_reference_avg)
            print("scores conv:", "source", score_conv_source_avg, "reference", score_conv_reference_avg, "summation",
          score_conv_source_avg + score_conv_reference_avg)
