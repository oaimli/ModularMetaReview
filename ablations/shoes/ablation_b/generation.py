from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List
import random


def gpt4_prompting(review_fragments: List):
    prompt_format = open("prompts_generation/prompt_generation.txt").read()
    review_text = "\n".join(review_fragments)
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
                n=1
                )

            final_meta_review = output_dict.choices[0].message.content
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)

    print(final_meta_review)

    return final_meta_review


def meta_generation(categorization_pairs: List) -> str:
    review_fragments = []
    for pair in categorization_pairs:
        review_fragments.extend(pair["review_fragments"])
    random.shuffle(review_fragments)
    meta_review = gpt4_prompting(review_fragments)

    return meta_review


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    with open(f"../../../modular_llama3/shoes/amasum_shoes_reasoning_result_{model_name}.json") as f:
        test_samples = json.load(f)

    results = []
    for sample in tqdm(test_samples):
        categorization_pairs = sample["categorization_pairs"]
        sample["meta_review_generated"] = meta_generation(categorization_pairs)
        results.append(sample)
        # print(sample)

    print(len(results))
    with open(f"amasum_shoes_generation_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
