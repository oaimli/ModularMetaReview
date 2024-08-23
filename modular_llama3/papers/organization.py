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
    prompt_format = open(f"../../optimization/organization/prompts_scientific/prompt_{mode.lower()}_{facet.lower()}.txt").read()
    prompt_content = prompt_format.replace("{{input_document}}", input_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "You are requested to do some extraction work. You must output the answer following the format of the example output, without any other useless content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=10
                )

            all_candidates = []
            all_candidates_len = []
            tmp = []
            for choice in output_dict.choices:
                output_content = choice.message.content
                if "no related fragments" not in output_content.lower():
                    content_len = len(output_content.split())
                    all_candidates_len.append(content_len)
                    tmp.append(content_len)
                    all_candidates.append(output_content.split("\n"))
            if len(all_candidates) < 5:
                outputs = []
            else:
                tmp.sort()
                outputs = all_candidates[all_candidates_len.index(tmp[int(len(tmp)/2)])]
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)

    return outputs


def categorizing_review(reviews: List) -> List:
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
            tmp[facet] = gpt4_prompting(review, facet, "review")
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
    facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]

    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    test_samples = []
    with jsonlines.open("../../datasets/peermeta_test.jsonl") as reader:
        for line in reader:
            test_samples.append(line)

    results = []
    # test_samples = random.sample(list(test_samples.keys()), 5)
    for test_sample in tqdm(test_samples):
        reviews = test_sample["source_documents"]
        test_sample["review_categorization"] = categorizing_review(reviews)
        results.append(test_sample)
        # print(sample)

    print(len(results))
    nlp = spacy.load("en_core_web_sm")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    for sample_index, sample in enumerate(results):
        reviews = sample["source_documents"]
        review_categorization = sample["review_categorization"]
        review_categorization_new = []
        for review, categorization in zip(reviews, review_categorization):
            review_categorization_new.append(matching_fragments(review, categorization))
        results[sample_index]["review_categorization"] = review_categorization_new

    with open(f"peermeta_categorization_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
