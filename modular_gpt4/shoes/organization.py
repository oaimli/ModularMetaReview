import random
import jsonlines
from openai import OpenAI
import time
from tqdm import tqdm
from typing import Dict, List
import json
import spacy
from rouge_score import rouge_scorer


def gpt4_prompting(input_text: str, facet: str, mode: str = "meta"):
    prompt_format = open(f"prompts_organization/prompt_{mode.lower()}_{facet.lower()}.txt").read()
    prompt_content = prompt_format.replace("{{input_document}}", input_text)
    # print(prompt_format)
    outputs = None
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system", "content": "You are requested to do some extraction work. You must produce the answer following the format of the example output, without other useless content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=8
                )

            for choice in output_dict.choices:
                # two requirements, following the jsonlines format and using the required key
                output_content = choice.message.content
                print(output_content)
                with open("output_tmp.jsonl", "w") as f:
                    f.write(output_content.strip())

                tmp = []
                try:
                    with jsonlines.open("output_tmp.jsonl") as reader:
                        for line in reader:
                            tmp.append(line)
                    output_keys = {[]}
                    for output in outputs:
                        output_keys.update(output.keys())
                    if len(output_keys.union({["extracted_fragment"]})) <= 1:
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
    output_keys = {[]}
    for output in outputs:
        output_keys.update(output.keys())
    assert len(output_keys.union({["extracted_fragment"]})) <= 1

    fragments = []
    for line in outputs:
        fragments.append(line["extracted_fragment"])

    return fragments


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


# Post-process the extraction results of models to make sure that extracted fragments are exactly the same as texts in the original document
def matching_fragments(document, facet_fragments):
    # trace judgements back to the original document, and return the corresponding fragments
    sentences = []
    for sent in nlp(document).sents:
        sentences.append(sent.text)

    result = {}
    for facet, fragments in facet_fragments.items():
        fragments_new = set([])
        for fragment in fragments:
            if fragment in document:
                if len(fragment) > 10:
                    fragments_new.add(fragment)
            else:
                fragment_sentences = []
                for sent in nlp(fragment).sents:
                    fragment_sentences.append(sent.text)
                for fragment_sentence in fragment_sentences:
                    if len(fragment_sentence) > 10:
                        if fragment_sentence in document:
                            fragments_new.add(fragment_sentence)
                        else:
                            rouges = []
                            for sentence in sentences:
                                scores = scorer.score(fragment_sentence, sentence)
                                rouges.append(scores["rouge2"].fmeasure + scores["rouge1"].fmeasure + scores[
                                    "rougeLsum"].fmeasure)
                            fragments_new.add(sentences[rouges.index(max(rouges))])
        result[facet] = list(fragments_new)

    return result


if __name__ == "__main__":
    random.seed(42)
    nlp = spacy.load("en_core_web_sm")
    facets = ["Breathability", "Durability", "Weight", "Cushioning", "Stability", "Flexibility", "Traction", "SizeFit", "Comfort", "Misc"]

    model_name = "gpt_4o"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    test_samples = []
    with jsonlines.open("../../datasets/amasum_shoes_test.jsonl") as reader:
        for line in reader:
            test_samples.append(line)

    results = []
    # test_samples = random.sample(list(test_samples.keys()), 5)
    for test_sample in tqdm(test_samples):
        reviews = test_sample["reviews"]
        test_sample["review_categorization"] = categorizing_review(reviews)
        results.append(test_sample)
        # print(sample)

    print(len(results))

    nlp = spacy.load("en_core_web_sm")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    for sample_index, sample_value in enumerate(results):
        meta_review = sample_value["meta_review"]
        meta_review_categorization = sample_value["meta_review_categorization"]
        results[sample_index]["meta_review_categorization"] = matching_fragments(meta_review,
                                                                                 meta_review_categorization)

        reviews = sample_value["reviews"]
        review_categorization = sample_value["review_categorization"]
        review_categorization_new = []
        for review, categorization in zip(reviews, review_categorization):
            review_categorization_new.append(matching_fragments(review["comment"], categorization))
        results[sample_index]["review_categorization"] = review_categorization_new

    with open(f"amasum_shoes_categorization_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
