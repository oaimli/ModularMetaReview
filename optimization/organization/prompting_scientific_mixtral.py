import random
import numpy as np
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import Dict, List
import spacy


def mixtral_prompting(input_text: str, facet: str, mode: str = "meta"):
    print("#######")
    prompt_format = open(f"prompts_scientific/prompt_{mode.lower()}_{facet.lower()}.txt").read()
    prompt_content = prompt_format.replace("{{input_document}}", input_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[
                    {"role": "system", "content": "You are requested to do some extraction work. You must output the answer following the format of the example output, without any other useless content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=10
                )

            all_candidates = []
            all_candidates_len = []
            for choice in output_dict.choices:
                output_content = choice.message.content
                if "no related fragments" not in output_content.lower():
                    all_candidates_len.append(len(output_content.split()))
                    all_candidates.append(output_content.split("\n"))
            if len(all_candidates) < 5:
                outputs = []
            else:
                outputs = all_candidates[all_candidates_len.index(int(np.median(all_candidates_len)))]
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
        result[facet] = mixtral_prompting(meta_review, facet, "meta")

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
            tmp[facet] = mixtral_prompting(review["comment"], facet, "review")
        result.append(tmp)

    return result


if __name__ == "__main__":
    random.seed(42)
    nlp = spacy.load("en_core_web_sm")
    facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]

    model_name = "mixtral8x7b_v01"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

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
