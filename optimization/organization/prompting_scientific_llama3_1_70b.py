# Inference with Transformers
import os
import random
import json
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import torch


def parsing_result(output):
    output = output.strip()
    print(f"######\n{output}######")
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
    print(f"Categorizing {mode}")
    prompt_format = open(f"prompts_scientific_llama3/prompt_{mode.lower()}_{facet.lower()}.txt").read()
    prompt_content = prompt_format.replace("{{input_document}}", input_text)
    messages = [
        [
            {"role": "system", "content": "Always answer with texts in a JSON Lines format, no other content."},
            {"role": "user", "content": prompt_content}
        ]
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        ).to(model.device)
    print(f"Running generation, and in the input there are {len(input_ids)} tokens")

    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        attention_mask=attention_mask,
        )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    output = parsing_result(response)
    fragments = []
    for line in output:
        if isinstance(line, dict) and "extracted_fragment" in line.keys():
            fragments.append(line["extracted_fragment"])
    print(fragments)
    return fragments



if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]

    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )

    # load the dataset
    dataset_path = "/home/miao4/punim0521/ModularMetaReview/annotations/scientific_reviews/annotation_data_small.json"
    random.seed(42)
    with open(dataset_path) as f:
        test_samples = json.load(f)

    random_keys = random.sample(list(test_samples.keys()), 5)
    print("all test data", len(random_keys))

    # generation
    results = {}
    for key in tqdm(random_keys):
        print(key)
        sample = test_samples[key]
        reviews = sample["reviews"]
        meta_review = sample["meta_review"]

        categorized_reviews = []
        for review in reviews:
            tmp = {}
            for facet in facets:
                tmp[facet] = llama_prompting(review["comment"], facet, "review")
            categorized_reviews.append(tmp)
        sample["review_categorization"] = categorized_reviews

        categorized_meta_review = {}
        for facet in facets:
            categorized_meta_review[facet] = llama_prompting(meta_review, facet, "meta")
        sample["meta_review_categorization"] = categorized_meta_review

        results[key] = sample

    # save generation results into json file
    output_file = "scientific_categorization_result_llama3_1_70b.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved to ", output_file)
