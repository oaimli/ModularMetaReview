from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def gpt4_prompting(review_fragments: List):
    prompt_format = open("prompts_reasoning/prompt_reasoning.txt").read()
    review_text = "\n".join(review_fragments)
    prompt_content = prompt_format.replace("{{review_fragments}}", review_text)
    print(len(review_fragments), len(review_text.split()), len(prompt_content.split()))
    # print(prompt_content)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "Always answer with only the summary, without other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=1
                )
            # all_candidates = []
            # all_candidates_len = []
            # tmp = []
            # for choice in output_dict.choices:
            #     output_content = choice.message.content
            #     content_len = len(output_content.split())
            #     all_candidates_len.append(content_len)
            #     tmp.append(content_len)
            #     all_candidates.append(output_content)
            # tmp.sort()
            # meta_generated = all_candidates[all_candidates_len.index(tmp[int(len(tmp) / 2)])]
            meta_generated = output_dict.choices[0].message.content
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(meta_generated)
    return meta_generated


def facet_reasoning(categorization_pairs: List) -> List:
    result = []
    for pair in categorization_pairs:
        print(pair["facet"])
        review_fragments = pair["review_fragments"]
        if len(review_fragments) == 0:
            pair["meta_generated"] = ""
        else:
            pair["meta_generated"] = gpt4_prompting(review_fragments)
        result.append(pair)

    return result


if __name__ == "__main__":
    model_name = "llama31_8b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    with open(f"amasum_shoes_selection_result_{model_name}.json") as f:
        test_samples = json.load(f)

    results = []
    for i, sample in tqdm(enumerate(test_samples)):
        # if i==30:
        categorization_pairs = sample["categorization_pairs"]
        sample["categorization_pairs"] = facet_reasoning(categorization_pairs)
        results.append(sample)
        # print(sample)

    print(len(results))
    with open(f"amasum_shoes_reasoning_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
