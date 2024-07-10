# llama3 installment is required, https://github.com/meta-llama/llama3/tree/main
import json
from typing import Dict, List
import spacy
import os
import random
import jsonlines
from tqdm import tqdm
from llama import Llama


def parsing_result(output):
    with open("output_tmp.jsonl", "w") as f:
        f.write(output.strip())
    results = []
    try:
        with jsonlines.open("output_tmp.jsonl") as reader:
            for line in reader:
                results.append(line)
    except jsonlines.InvalidLineError as err:
        print("Jsonlines parsing error,", err)
    return results


def llama_prompting(input_text: str, facet: str, mode: str = "meta"):
    sentences = []
    for sent in nlp(input_text).sents:
        sentences.append(sent.text)
    random_positions = [random.randint(0, len(sentences)) for _ in range(5)]
    random_nums = [random.randint(1, 6) for _ in range(3)]
    example_output = []
    for position, num in zip(random_positions, random_nums):
        tmp = " ".join(sentences[position: position + num])
        example_output.append({"extracted_fragment": tmp.strip()})
    with jsonlines.open("example_tmp.jsonl", "w") as writer:
        writer.write_all(example_output)
    with open("example_tmp.jsonl", "r") as f:
        example_output_text = f.read()

    prompt_format = open(f"prompts_scientific/prompt_{mode.lower()}_{facet.lower()}.txt").read()
    prompt_content = prompt_format.replace("{{input_document}}", input_text).replace("{{example_output}}",
                                                                                     example_output_text)
    print("tokens", len(generator.formatter.tokenizer.encode(prompt_content, bos=True, eos=True)))
    with open("prompt_tmp.txt", "w") as f:
        f.write(prompt_content)

    messages = [
        [
            {"role": "user",
             "content": prompt_content}
        ]
    ]
    print("Running generation, and in the input there are ")
    result = generator.chat_completion(
        messages,
        max_gen_len=1024,
        temperature=0.7,
        top_p=0.92,
        )[0]

    output = result["generation"]["content"]
    print(output)
    fragments = []
    for line in output:
        if isinstance(line, dict) and "extracted_fragment" in line.keys():
            fragments.append(line["extracted_fragment"])
    print(fragments)
    return fragments


def categorizing_meta_review(meta_review: str) -> Dict:
    """
    Args:
        meta_review: the meta-review of a sample
    Returns:
        result: a dictionary of extracted fragments for different review facets
    """
    result = {}
    for facet in facets:
        result[facet] = llama_prompting(meta_review, facet, "meta")

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
            tmp[facet] = llama_prompting(review["comment"], facet, "review")
        result.append(tmp)
    return result


if __name__ == "__main__":
    # Run with torchrun --nproc_per_node 4 prompting_scientific_llama3.py

    nlp = spacy.load("en_core_web_sm")
    facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]

    model_name = "llama3"
    ckpt_dir = "/data/projects/punim0521/tmp/Meta-Llama-3-70B-Instruct-four-nodes/"
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.model")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=8192,
        max_batch_size=1
        )
    print("Model loading done")

    with open("../../annotations/scientific_reviews/annotation_data_small.json") as f:
        test_samples = json.load(f)

    results = {}
    random_samples = random.sample(list(test_samples.keys()), 5)
    for key in tqdm(random_samples):
        print(key)
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
