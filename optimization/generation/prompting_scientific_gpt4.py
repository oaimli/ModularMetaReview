from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def gpt4_prompting(metas_generated: List):
    prompt_format = open("prompt_scientific_gpt4.txt").read()
    review_text = "\n".join(metas_generated)
    prompt_content = prompt_format.replace("{{metas_generated}}", review_text)
    # print(prompt_format)
    final_meta_review = ""
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system", "content": "Always answer with only the summary, no other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=5
                )
            for choice in output_dict.choices:
                tmp = choice.message.content
                if len(tmp) > 0:
                    final_meta_review = tmp
                    break
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

    meta_review = gpt4_prompting(metas_generated)

    return meta_review


if __name__ == "__main__":
    model_name = "gpt_4o"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    with open("../reasoning/scientific_reasoning_result_gpt_4o.json") as f:
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
