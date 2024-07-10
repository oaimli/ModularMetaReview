import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def parsing_result(output):
    with open("output_tmp.jsonl", "w") as f:
        f.write(output.strip())
    tmp = []
    try:
        with jsonlines.open("output_tmp.jsonl") as reader:
            for line in reader:
                tmp.append(line)
    except jsonlines.InvalidLineError as err:
        print("Jsonlines parsing error,", err)
    return tmp


def gpt4_prompting(review_fragments: List, facet: str):
    prompt_format = open("prompt_gpt4.txt").read()
    review_text = "\n".join(review_fragments)
    prompt_content = prompt_format.replace("{{review_fragments}}", review_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Always answer with only the summary in JSON, no other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=5
                )
            output = []
            for choice in output_dict.choices:
                tmp = parsing_result(choice.message.content)
                if len(tmp) > 0:
                    output = tmp
                    break
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(output)
    meta_generated = ""
    if len(output) > 0:
        if "summary" in output[0].keys():
            meta_generated = output[0]["summary"]
    print(meta_generated)

    return meta_generated


def facet_reasoning(categorization_pairs: List) -> List:
    result = []
    for pair in categorization_pairs:
        review_fragments = pair["review_fragments"]
        facet = pair["facet"]
        pair["meta_generated"] = gpt4_prompting(review_fragments, facet)
        result.append(pair)

    return result


if __name__ == "__main__":
    model_name = "gpt4"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    with open("../selection/scientific_selection_result_llama3_70b.json") as f:
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
