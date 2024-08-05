from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from arguments import ModelArguments, DataArguments, TrainingArguments
import jsonlines
import random
from datasets import load_dataset
import os
from tqdm import tqdm

def predict(
        model_args,
        data_args,
        training_args
):
    # load model and tokenizer
    if training_args.resume_from_checkpoint != None:
        model_in_use = training_args.resume_from_checkpoint
    else:
        model_in_use = model_args.model_name_or_path
    print("loading model:", model_in_use)

    tokenizer_class = LlamaTokenizer if "llama" in model_args.model_name_or_path else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        model_in_use,
        padding_side="right",
        use_fast=True,
    )
    print("tokenizer bos", tokenizer.bos_token, tokenizer.bos_token_id)
    print("tokenizer eos", tokenizer.eos_token, tokenizer.eos_token_id)
    print("tokenizer pad", tokenizer.pad_token, tokenizer.pad_token_id)
    print("tokenizer unk", tokenizer.unk_token, tokenizer.unk_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_in_use, trust_remote_code=True
    )
    print(model.config)

    # data_module = get_data_module(tokenizer=tokenizer, prompt_format=model_args.prompt_format, data_args=data_args,
    #                               model_args=model_args)
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, data_collator=data_module["data_collator"])
    # evaluation_results = trainer.evaluate(eval_dataset=data_module["eval_dataset"], metric_key_prefix="eval")
    # print(evaluation_results)
    # test_results = trainer.evaluate(eval_dataset=data_module["test_dataset"], metric_key_prefix="test")
    # print(test_results)

    # generate
    model.to("cuda")
    model.eval()
    all_data = load_dataset('json', data_files=data_args.dataset_path + '%s.jsonl' % data_args.dataset_name,
                            split='all')
    test_data = all_data.filter(lambda s: s['label'] == 'test')
    if data_args.num_test_samples > 0:
        random.seed(42)
        test_data = test_data.select(random.choices(range(len(test_data)), k=data_args.num_test_samples))
    print("all test data", len(test_data))

    results = []
    contexts = []
    conversation_histories = []
    predictions = []
    for sample in tqdm(test_data):
        conversation_history = " ".join(sample["conversation_history"][-10:])
        knowledge_source = sample["knowledge_source"]

        conversation_history_tokenized = tokenizer.encode(conversation_history, truncation=False)
        knowledge_source_tokenized = tokenizer.encode(knowledge_source, max_length=tokenizer.model_max_length - len(
            conversation_history_tokenized), truncation=True)
        source_text = tokenizer.decode(knowledge_source_tokenized,
                                       skip_special_tokens=True) + " " + tokenizer.bos_token + " " + tokenizer.decode(
            conversation_history_tokenized, skip_special_tokens=True)

        # the bos token will be added here, and truncation is not necessary here
        input_dict = tokenizer(
            [source_text],
            return_tensors="pt",
            truncation=False
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
        predicted_response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        sample["predicted_response"] = predicted_response
        results.append(sample)

        contexts.append(knowledge_source)
        conversation_histories.append(conversation_history)
        predictions.append(predicted_response)

    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    output_file = training_args.output_dir + "/%s.jsonl" % training_args.output_file
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(list(results))

    print(unieval_dialogue(conversation_histories=conversation_histories, contexts=contexts, predictions=predictions))


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_predict:
        predict(model_args, data_args, training_args)
