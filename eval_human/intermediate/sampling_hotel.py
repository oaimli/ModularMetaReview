import json
import random
import time
import jsonlines
import numpy as np
import spacy
from typing import List
from rouge_score import rouge_scorer
from openai import OpenAI


random.seed(42)

# filter out samples that are used in the pair-wise human evaluation on meta-reviews
with open("../meta_reviews/generations_space.json") as f:
    samples_meta_reviews = json.load(f)
indexes_meta_reviews = []
for sample_index in samples_meta_reviews.keys():
    indexes_meta_reviews.append(int(sample_index.split("_")[1]))
print(indexes_meta_reviews)

# load the original test dataset
samples_test = []
with jsonlines.open("../../datasets/space_test.jsonl") as reader:
    for line in reader:
        samples_test.append(line)

samples_all = []
for sample_test in samples_test:
    meta_reviews = sample_test["gold_summaries_general"]
    sources = sample_test["source_documents"]

    sample_new = {}
    sample_new["source_documents"] = random.sample(sources, 10)
    # human-written reference
    sample_new["generation_decomposed"] = ""
    sample_new["steps_decomposed"] = ""
    sample_new["generation_modular"] = ""
    sample_new["steps_modular"] = ""
    samples_all.append(sample_new)

sampled_indexes = []
for sample_index, sample in enumerate(samples_all):
    if sample_index not in indexes_meta_reviews:
        sampled_indexes.append(sample_index)
sampled_indexes = random.sample(sampled_indexes, 10)
print(sampled_indexes)

# get the samples from the test dataset
samples_sampled = {}
for sample_index, sample in enumerate(samples_all):
    if sample_index in sampled_indexes:
        samples_sampled[f"index_{sample_index}"] = sample
print("Sampled samples", len(samples_sampled))

# statistics of these sampled samples
source_lengths = []
for sample_key, sample in samples_sampled.items():
    # print(sample_key)
    source_documents = sample["source_documents"]
    source_lengths.append(len(" ".join(source_documents).split()))
print("Average source length", np.mean(source_lengths))

# get generated meta-reviews and categorization pairs with our modular approach, same as in modular_llama3
def get_generations_with_modular(samples_sampled):
    random.seed(42)
    nlp = spacy.load("en_core_web_sm")
    facets = ["Building", "Cleanliness", "Food", "Location", "Rooms", "Service"]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    def prompting_categorization(input_text: str, facet: str, mode: str = "meta"):
        prompt_format = open(f"../../modular_llama3/hotels/prompts_organization/prompt_{mode.lower()}_{facet.lower()}.txt").read()
        prompt_content = prompt_format.replace("{{input_document}}", input_text)
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to do some extraction work. You must output the answer following the format of the example output, without any other useless content."},
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
                    outputs = all_candidates[all_candidates_len.index(tmp[int(len(tmp) / 2)])]
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
                tmp[facet] = prompting_categorization(review, facet, "review")
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

    def prompting_reasoning(review_fragments: List):
        prompt_format = open("../../modular_llama3/hotels/prompts_reasoning/prompt_reasoning.txt").read()
        review_text = "\n".join(review_fragments)
        prompt_content = prompt_format.replace("{{review_fragments}}", review_text)
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system", "content": "Always answer with only the summary, without other content."},
                        {"role": "user",
                         "content": prompt_content}
                        ],
                    n=8
                    )
                all_candidates = []
                all_candidates_len = []
                tmp = []
                for choice in output_dict.choices:
                    output_content = choice.message.content
                    content_len = len(output_content.split())
                    all_candidates_len.append(content_len)
                    tmp.append(content_len)
                    all_candidates.append(output_content)
                tmp.sort()
                meta_generated = all_candidates[all_candidates_len.index(tmp[int(len(tmp) / 2)])]
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
        print(meta_generated)
        return meta_generated

    def facet_reasoning(categorization_pairs: List) -> List:
        result = []
        for pair in categorization_pairs:
            review_fragments = pair["review_fragments"]
            if len(review_fragments) == 0:
                pair["meta_generated"] = ""
            else:
                pair["meta_generated"] = prompting_reasoning(review_fragments)
            result.append(pair)

        return result

    def prompting_final(metas_generated: List):
        prompt_format = open("../../modular_llama3/hotels/prompts_generation/prompt_generation.txt").read()
        review_text = "\n".join(metas_generated)
        prompt_content = prompt_format.replace("{{metas_generated}}", review_text)
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "Always answer with only the predicted summary, no other content."},
                        {"role": "user",
                         "content": prompt_content}
                        ],
                    n=8
                    )
                all_candidates = []
                all_candidates_len = []
                tmp = []
                for choice in output_dict.choices:
                    output_content = choice.message.content
                    content_len = len(output_content.split())
                    all_candidates_len.append(content_len)
                    tmp.append(content_len)
                    all_candidates.append(output_content)
                tmp.sort()
                final_meta_review = all_candidates[all_candidates_len.index(tmp[int(len(tmp) / 2)])]
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
            if pair["meta_generated"] != "":
                metas_generated.append(pair["meta_generated"])

        meta_review = prompting_final(metas_generated)

        return meta_review

    for sample_key, sample in samples_sampled.items():
        reviews = sample["source_documents"]

        # extract text fragments from each review
        reviews_categorization = categorizing_review(reviews)
        tmp = []
        for review, categorization in zip(reviews, reviews_categorization):
            tmp.append(matching_fragments(review, categorization))
        reviews_categorization = tmp

        # group extracted fragments based on review aspects
        categorization_pairs = []
        for facet in reviews_categorization[0].keys():
            tmp = {}
            tmp["aspect"] = facet

            review_fragments = []
            for review_categorization in reviews_categorization:
                review_fragments.extend(review_categorization[facet])
            tmp["review_fragments"] = review_fragments
            categorization_pairs.append(tmp)

        # generate aspect-focused meta-review for each review aspect
        categorization_pairs = facet_reasoning(categorization_pairs)

        # generate the final meta-review with aggregating all aspect-focused meta-reviews
        meta_review = meta_generation(categorization_pairs)

        sample["generation_modular"] = meta_review
        sample["steps_modular"] = categorization_pairs
        samples_sampled[sample_key] = sample

    return samples_sampled

# get generated meta-reviews and intermediate steps with decomposed prompting, same as in llama3_pr/hotels
def get_generations_with_decomposed(samples_sampled):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    for sample_key, sample in samples_sampled.items():
        print("Processing", sample_key)
        source_documents = sample["source_documents"]
        source_text = "\n".join(source_documents)
        prompt_content = f"Please give me sequential steps to write a summary specific for the following reviews on a hotel.\n\n Reviews on a hotel:\n {source_text}\n\nThe steps to write a summary in different lines:"
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to write the steps. Please output the final answer with only the steps in different lines, no other useless content."},
                        {"role": "user",
                         "content": prompt_content}
                        ],
                    n=1
                    )
                actions = output_dict.choices[0].message.content
                break
            except Exception as e:
                if "limit" in str(e):
                    time.sleep(2)

        decomposed_steps = []
        actions_list = actions.split("\n")
        output = ""
        for j, action in enumerate(actions_list):
            step = {"action": action, "output": ""}
            if j == 0:
                prompt_content = f"{source_text}\nPlease follow the instruction below and give your output.\n {action}\nThe output:"
            else:
                prompt_content = f"{output}\nPlease follow the instruction below and give your output.\n {action}\nThe output:"
            # print(prompt_format)
            while True:
                try:
                    output_dict = client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                        messages=[
                            {"role": "system",
                             "content": "You are requested to follow the instruction and only generate the requested output."},
                            {"role": "user",
                             "content": prompt_content}
                            ],
                        n=1
                        )
                    output = output_dict.choices[0].message.content
                    break
                except Exception as e:
                    if "limit" in str(e):
                        time.sleep(2)
            step["output"] = output
            print(step)
            decomposed_steps.append(step)

        sample["generation_decomposed"] = output # the output of the last step
        sample["steps_decomposed"] = decomposed_steps
        samples_sampled[sample_key] = sample

    return samples_sampled


samples_sampled = get_generations_with_modular(samples_sampled)
samples_sampled = get_generations_with_decomposed(samples_sampled)
with open("sampled_hotel.json", "w") as f:
    json.dump(samples_sampled, f, indent=4)
