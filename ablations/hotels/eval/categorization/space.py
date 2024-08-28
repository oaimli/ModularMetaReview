import json
from openai import OpenAI
from typing import Dict
import time
from tqdm import tqdm


def llama3_prompting(input_text: str, facet: str, mode: str = "meta"):
    # print("#######")
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

    with open("../info.json") as f:
        info = json.load(f)

    dataset_name = "space"
    print(dataset_name)
    facets = ["Building", "Cleanliness", "Food", "Location", "Rooms", "Service"]
    prompt_folder = "../../modular_llama3/hotels/prompts_organization"

    generations_info = info[dataset_name]
    for generation_info in tqdm(generations_info):
        generation_file = generation_info["generation_file"]
        print(generation_file)
        output_name = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
        print(output_name)
        candidate_key = generation_info["candidate_key"]
        reference_key = generation_info["reference_key"]
        source_key = "source_documents"
        with open(generation_file) as f:
            samples = json.load(f)

        for i, sample in enumerate(samples):
            candidate = sample[candidate_key]
            if isinstance(sample[reference_key], str):
                reference = sample[reference_key]
            else:
                reference = sample[reference_key][0]  # SPACE has multiple references

            categorization_reference = categorizing_meta_review(reference)
            categorization_candidate = categorizing_meta_review(candidate)
            sample["categorization_reference"] = categorization_reference
            sample["categorization_candidate"] = categorization_candidate
            samples[i] = sample

        with open(output_name, "w") as f:
            json.dump(samples, f, indent=4)

