from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def llama3_prompting(metas_generated: List):
    prompt_format = open("prompt_generation_scientific.txt").read()
    review_text = "\n".join(metas_generated)
    prompt_content = prompt_format.replace("{{metas_generated}}", review_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "Always answer with only the predicted summary, no other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=8
                )
            all_candidates = []
            all_candidates_len = []
            tmp = []
            for choice in output_dict.choices:
                output_content = choice.message.content
                content_len = len(output_content.split())
                all_candidates_len.append(content_len)
                tmp.append(content_len)
                all_candidates.append(output_content)
            tmp.sort()
            final_meta_review = all_candidates[all_candidates_len.index(tmp[int(len(tmp) / 2)])]
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)

    print(final_meta_review)

    return final_meta_review


def meta_generation(categorization_pairs: List) -> str:
    metas_generated = []
    for pair in categorization_pairs:
        metas_generated.append(pair["meta_generated"])

    meta_review = llama3_prompting(metas_generated)

    return meta_review


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    with open("scientific_reasoning_result_llama31_70b.json") as f:
        test_samples = json.load(f)

    results = {}
    for key in tqdm(test_samples):
        sample = test_samples[key]
        categorization_pairs = sample["categorization_pairs"]
        sample["meta_review_generated"] = meta_generation(categorization_pairs)
        results[key] = sample
        # print(sample)

    print(len(results))
    with open(f"scientific_generation_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
