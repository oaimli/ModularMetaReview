from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def llama3_prompting(review_fragments: List):
    prompt_format = open("prompt_reasoning_scientific.txt").read()
    review_text = "\n".join(review_fragments)
    prompt_content = prompt_format.replace("{{review_fragments}}", review_text)
    # print(prompt_format)
    meta_generated = ""
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "Always answer with only the summary, without other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=5
                )
            for choice in output_dict.choices:
                tmp = choice.message.content
                if len(tmp.split()) > 8:
                    meta_generated = tmp
                    break
            if meta_generated != "":
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(output)
    return meta_generated


def facet_reasoning(categorization_pairs: List) -> List:
    result = []
    for pair in categorization_pairs:
        review_fragments = pair["review_fragments"]
        pair["meta_generated"] = llama3_prompting(review_fragments)
        result.append(pair)

    return result


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    with open("../selection/scientific_selection_result_llama31_70b.json") as f:
        test_samples = json.load(f)

    results = {}
    for key in tqdm(test_samples):
        sample = test_samples[key]
        categorization_pairs = sample["categorization_pairs"]
        sample["categorization_pairs"] = facet_reasoning(categorization_pairs)
        results[key] = sample
        # print(sample)

    print(len(results))
    with open(f"scientific_reasoning_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
