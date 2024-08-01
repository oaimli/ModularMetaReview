# Inference with the official code from Meta LLaMA-3
import os
import random
import json
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import spacy
import torch


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Name or path of the pre-trained model."})
    max_length_model: Optional[int] = field(default=8192, metadata={"help": "Max input length of the model."})
    # predict with sampling or contrastive search
    max_predict_length: Optional[int] = field(default=512, metadata={
        "help": "Max predicted target length when generation, excluding the source part."})
    temperature: Optional[float] = field(default=0.6,
                                         metadata={"help": "The value to modulate the next token probabilities."})
    top_p: Optional[float] = field(default=0.9, metadata={
        "help": "most probable tokens with probabilities that add up to top_p or higher are kept for generation"})
    output_file: str = field(default="generation.json", metadata={"help": "The name of the output file."})
    max_batch_size: Optional[int] = field(default=1, metadata={"help": "The maximum batch size for inference."})


@dataclass
class DataArguments:
    dataset_path: str = field(default=None, metadata={"help": "Path to the test dataset."})
    num_test_samples: int = field(default=-1, metadata={"help": "Number of test samples."})


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
        max_new_tokens=model_args.max_predict_length,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=model_args.temperature,
        top_p=model_args.top_p,
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

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )

    # load the dataset
    random.seed(42)
    with open(data_args.dataset_path) as f:
        test_samples = json.load(f)

    random_keys = random.sample(list(test_samples.keys()), data_args.num_test_samples)
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
    with open(model_args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved to ", model_args.output_file)
