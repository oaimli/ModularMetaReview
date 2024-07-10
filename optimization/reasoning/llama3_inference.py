# Inference with the official code from Meta LLaMA-3
import os
import json
import jsonlines
from tqdm import tqdm
from llama import Llama
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Name or path of the pre-trained model."})
    max_length_model: Optional[int] = field(default=8192, metadata={"help": "Max input length of the model."})
    # predict with sampling or contrastive search
    max_predict_length: Optional[int] = field(default=512, metadata={
        "help": "Max predicted target length when generation, excluding the source part."})
    temperature: Optional[float] = field(default=0.7,
                                         metadata={"help": "The value to modulate the next token probabilities."})
    top_p: Optional[float] = field(default=0.92, metadata={
        "help": "most probable tokens with probabilities that add up to top_p or higher are kept for generation"})
    output_file: str = field(default="generation.json", metadata={"help": "The name of the output file."})
    max_batch_size: Optional[int] = field(default=1, metadata={"help": "The maximum batch size for inference."})


@dataclass
class DataArguments:
    dataset_path: str = field(default=None, metadata={"help": "Path to the test dataset."})


def parsing_result(output):
    output_stripped = output.strip()
    print(f"######\n{output}######")
    with open("output_tmp.jsonl", "w") as f:
        f.write(output_stripped)
    tmp = []
    try:
        with jsonlines.open("output_tmp.jsonl") as reader:
            for line in reader:
                tmp.append(line)
    except jsonlines.InvalidLineError as err:
        print("Jsonlines parsing error,", err)
    return tmp


def llama_prompting(review_fragments: List, facet: str):
    prompt_format = open("prompt_llama3.txt").read()
    review_text = "\n".join(review_fragments)
    prompt_content = prompt_format.replace("{{review_fragments}}", review_text)
    messages = [
        [
            {"role": "system", "content": "Always answer with only the summary in JSON, no other content."},
            {"role": "user", "content": prompt_content}
        ]
    ]
    tokens_num = len(generator.formatter.tokenizer.encode(prompt_content, bos=True, eos=True))
    print(f"Running generation, and in the input there are {tokens_num} tokens")
    print(prompt_content)
    # i = 0
    # while True:
    #     result = generator.chat_completion(
    #         messages,
    #         max_gen_len=model_args.max_predict_length,
    #         temperature=model_args.temperature,
    #         top_p=model_args.top_p,
    #         )[0]
    #     output = parsing_result(result["generation"]["content"])
    #     if len(output) > 0 or i>9:
    #         break
    #     else:
    #         print("re-generating")
    #         i += 1
    result = generator.chat_completion(
        messages,
        max_gen_len=model_args.max_predict_length,
        temperature=model_args.temperature,
        top_p=model_args.top_p,
        )[0]
    output = parsing_result(result["generation"]["content"])
    meta_generated = ""
    if len(output) > 0:
        if "summary" in output[0].keys():
            meta_generated = output[0]["summary"]
    print(meta_generated)
    return meta_generated


def facet_reasoning(categorization_pairs: List) -> List:
    result = []
    for pair in categorization_pairs:
        review_fragments = pair["review_fragments"]
        facet = pair["facet"]
        pair["meta_generated"] = llama_prompting(review_fragments, facet)
        result.append(pair)

    return result


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    ckpt_dir = model_args.model_name_or_path
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.model")

    # load the model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=model_args.max_length_model,
        max_batch_size=model_args.max_batch_size,
    )

    # load the dataset
    with open(data_args.dataset_path) as f:
        test_samples = json.load(f)
    print("all test data", len(test_samples))

    # generation
    results = {}
    for key in tqdm(test_samples.keys()):
        print(key)
        sample = test_samples[key]
        categorization_pairs = sample["categorization_pairs"]
        sample["categorization_pairs"] = facet_reasoning(categorization_pairs)
        results[key] = sample

    # save generation results into json file
    with open(model_args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved to ", model_args.output_file)
