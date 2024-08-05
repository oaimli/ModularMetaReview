from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from arguments import ModelArguments, DataArguments, TrainingArguments
import jsonlines
import random
from datasets import load_dataset
import os
from tqdm import tqdm
import torch


def predict(
        model_args,
        data_args,
        training_args
        ):
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="right", use_fast=True)
    print("tokenizer bos", tokenizer.bos_token, tokenizer.bos_token_id)
    print("tokenizer eos", tokenizer.eos_token, tokenizer.eos_token_id)
    print("tokenizer pad", tokenizer.pad_token, tokenizer.pad_token_id)
    print("tokenizer unk", tokenizer.unk_token, tokenizer.unk_token_id)

    model = AutoModelForCausalLM.from_pretrained(model_args, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    print(model.config)

    test_data = load_dataset('json', data_files=data_args.dataset_path + '%s_test.jsonl' % data_args.dataset_name,
                            split='all')
    if data_args.num_test_samples > 0:
        random.seed(42)
        test_data = test_data.select(random.choices(range(len(test_data)), k=data_args.num_test_samples))
    print("all test data", len(test_data))

    results = []
    for sample in tqdm(test_data):
        source_text = ""
        for document in sample["source_documents"]:
            source_text = source_text + " " + document
        # the bos token will be added here, and truncation is not necessary here
        input_dict = tokenizer(
            [source_text],
            return_tensors="pt",
            truncation=True,
            max_length=model_args.max_length_model - model_args.max_predict_length
            )
        input_ids = input_dict.input_ids
        attention_mask = input_dict.attention_mask
        output_ids = model.generate(
            input_ids=input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            max_length=len(input_ids[0]) + model_args.max_predict_length,
            min_length=len(input_ids[0]) + model_args.min_predict_length,
            do_sample=model_args.do_sample,
            top_p=model_args.top_p,
            num_beams=model_args.num_beams,
            pad_token_id=tokenizer.eos_token_id
            )
        predicted_meta_review = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        sample["meta_review_generated"] = predicted_meta_review
        results.append(sample)

    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    output_file = training_args.output_dir + "/%s.jsonl" % training_args.output_file
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(list(results))


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_predict:
        predict(model_args, data_args, training_args)
