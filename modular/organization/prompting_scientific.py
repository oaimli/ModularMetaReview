import random
import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import Dict, List


def parsing_result(output):
    tmp = []
    for fragment in output.split("\n"):
        if fragment.strip() != "":
            tmp.append(fragment)
    return tmp


def gpt4_prompting(input_text: str, facet: str, mode: str = "meta"):
    prompt_format = open(f"prompts_scientific/prompt_{mode}_{facet}.txt").read()
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": prompt_format.replace("{{input_document}}", input_text)}
                    ],
                n=1
                )
            output = parsing_result(output_dict.choices[0].message.content)
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(output)
    return output


def llama_prompting(input_text: str, facet: str, mode: str = "meta"):
    return None


def mistral_prompting(input_text: str, facet: str, mode: str = "meta"):
    return None


def categorizing_meta_review(meta_review: str, model_name: str = "gpt4") -> Dict:
    """
    Args:
        meta_review: the meta-review of a sample
        model_name: the name of a model
    Returns:
        result: a dictionary of extracted fragments for different review facets
    """
    prompting = None
    if model_name == "gpt4":
        prompting = gpt4_prompting
    if model_name == "llama":
        prompting = llama_prompting
    if model_name == "mistral":
        prompting = mistral_prompting

    if prompting != None:
        result = {}
        for facet in facets:
            result[facet] = prompting(meta_review, facet, "meta")
    else:
        result = {}
        print("The model name is not correct.")

    return result


def categorizing_review(reviews: List[Dict], model_name: str) -> List:
    """
    Args:
        reviews: the list of reviews in the original dataset
        model_name: the name of a model
    Returns:
        result: a list of dictionaries
    """
    prompting = None
    if model_name == "gpt4":
        prompting = gpt4_prompting
    if model_name == "llama":
        prompting = llama_prompting
    if model_name == "mistral":
        prompting = mistral_prompting

    result = []
    if prompting != None:
        for review in reviews:
            tmp = {}
            for facet in facets:
                tmp[facet] = prompting(review["comment"], facet, "review")
            result.append(tmp)
    else:
        print("The model name is not correct.")

    return result


if __name__ == "__main__":
    facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]

    model_name = "gpt4"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    with open("../../annotations/scientific_reviews/annotation_data_small.json") as f:
        test_samples = json.load(f)

    results = {}
    for key, sample in tqdm(random.sample(test_samples.items(), 5)):
        reviews = sample["reviews"]
        meta_review = sample["meta_review"]
        sample["review_categorization"] = categorizing_review(reviews, model_name)
        sample["meta_review_categorization"] = categorizing_meta_review(meta_review, model_name)
        results[key] = sample
        # print(sample)

    print(len(results))
    with open(f"scientific_categorization_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
