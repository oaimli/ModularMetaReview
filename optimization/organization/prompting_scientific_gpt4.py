import random
import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import Dict, List
import spacy


def gpt4_prompting(input_text: str, facet: str, mode: str = "meta"):
    prompt_format = open(f"prompts_scientific/prompt_{mode.lower()}_{facet.lower()}.txt").read()
    prompt_content = prompt_format.replace("{{input_document}}", input_text)
    # print(prompt_format)
    outputs = None
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system", "content": "You are requested to do some extraction work. You must output the answer following the format of the example output, without any other useless content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=8
                )

            for choice in output_dict.choices:
                # two requirements, following the jsonlines format and using the required key
                output_content = choice.message.content
                print(output_content)
                if "no related fragments" in output_content.lower():
                    outputs = []
                    break

                with open("output_tmp.jsonl", "w") as f:
                    f.write(output_content.strip())
                tmp = []
                try:
                    with jsonlines.open("output_tmp.jsonl") as reader:
                        for line in reader:
                            tmp.append(line)
                    output_keys = set([])
                    for output in outputs:
                        output_keys.update(output.keys())
                    if len(output_keys.union({"extracted_fragment"})) <= 1:
                        outputs = tmp
                        break
                except jsonlines.InvalidLineError as err:
                    print("Jsonlines parsing error,", err)

            if outputs != None:
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)

    print(outputs)
    output_keys = set([])
    for output in outputs:
        output_keys.update(output.keys())
    assert len(output_keys.union({"extracted_fragment"})) <= 1

    fragments = []
    for line in outputs:
        fragments.append(line["extracted_fragment"])

    return fragments


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
        result[facet] = gpt4_prompting(meta_review, facet, "meta")

    return result


def categorizing_review(reviews: List[Dict]) -> List:
    """
    Args:
        reviews: the list of reviews in the original dataset
    Returns:
        result: a list of dictionaries
    """
    result = []
    for review in reviews:
        tmp = {}
        for facet in facets:
            tmp[facet] = gpt4_prompting(review["comment"], facet, "review")
        result.append(tmp)

    return result


if __name__ == "__main__":
    random.seed(42)
    nlp = spacy.load("en_core_web_sm")
    facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]

    model_name = "gpt_4o"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    with open("../../annotations/scientific_reviews/annotation_data_small.json") as f:
        test_samples = json.load(f)

    results = {}
    random_samples = random.sample(list(test_samples.keys()), 5)
    for key in tqdm(random_samples):
        sample = test_samples[key]
        reviews = sample["reviews"]
        meta_review = sample["meta_review"]
        sample["review_categorization"] = categorizing_review(reviews)
        sample["meta_review_categorization"] = categorizing_meta_review(meta_review)
        results[key] = sample
        # print(sample)

    print(len(results))
    with open(f"scientific_categorization_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
