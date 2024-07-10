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
    with open("prompt_tmp.txt", "w") as f:
        f.write(prompt_content)
    prompt_content = "Miao is interested in natural language processing (NLP) which aims to enable machines to read, reason and generate human language. Throughout the years of pursuing the PhD, his primary area of interest revolves around multi-document language generation, with a special focus on automatically summarizing multiple input documents. While automated natural language summarization has achieved significant progress and pre-trained language models have demonstrated the capability to generate plausible summaries, the effectiveness of consolidating information from multiple documents remains uncertain and largely unexplored when these models are asked to summarize a collection of documents. Miao’s PhD research aims to investigate multi-document summarization from the perspective of information consolidation and make it less opaque and more grounded. In the long term, Miao’s overarching research goal is to (1) understand how humans comprehend multi-source information with reasoning to make their decisions in language generation from first principles, (2) explore the potential of machines to achieve superhuman-level reasoning and consolidation over multiple sources with voluminous and complex heterogeneous information and realize human-like communication conveying information based on devised consequences from them, and (3) develop high-quality evaluations of artificial intelligence systems on complex natural language generation tasks."

    messages = [
        [
            {"role": "user",
             "content": prompt_content}
        ]
    ]
    tokens_num = len(generator.formatter.tokenizer.encode(prompt_content, bos=True, eos=True))
    print(f"Running generation, and in the input there are {tokens_num} tokens")
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
        max_batch_size=4
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
