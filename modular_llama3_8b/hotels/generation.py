from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def gpt4_prompting(metas_generated: List):
    prompt_format = open("prompts_generation/prompt_generation.txt").read()
    review_text = "\n".join(metas_generated)
    prompt_content = prompt_format.replace("{{metas_generated}}", review_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
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
        if pair["meta_generated"] != "":
            metas_generated.append(pair["meta_generated"])

    meta_review = gpt4_prompting(metas_generated)

    return meta_review


if __name__ == "__main__":
    model_name = "llama31_8b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    with open(f"space_reasoning_result_{model_name}.json") as f:
        test_samples = json.load(f)

    results = []
    for sample in tqdm(test_samples):
        categorization_pairs = sample["categorization_pairs"]
        sample["meta_review_generated"] = meta_generation(categorization_pairs)
        results.append(sample)
        # print(sample)

    print(len(results))
    with open(f"space_generation_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
