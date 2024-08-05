#!/usr/bin/env python3
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, LEDTokenizer, LEDConfig, LEDForConditionalGeneration
from transformers import EarlyStoppingCallback
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import wandb
import argparse


parser = argparse.ArgumentParser()
# General
parser.add_argument("--save_path", type=str, help="The path to save model checkpoints, logs and results")
parser.add_argument("--pretrained_model", type=str, default="allenai/led-large-16384",
                    help="The name of the pretrained model")
parser.add_argument("--data_path", type=str, default="../../datasets/")
parser.add_argument("--max_length_input", default=16384, type=int)
parser.add_argument("--max_length_tgt", default=512, type=int)
parser.add_argument("--min_length_tgt", default=0, type=int)
parser.add_argument("--save_top_k", default=1, type=int)
parser.add_argument("--dataset_name", type=str, default="peermeta")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for dataloader")
parser.add_argument("--batch_size", default=4, type=int)

parser.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing to save memory")
parser.add_argument("--rand_seed", type=int, default=42,
                    help="Seed for random sampling, useful for few shot learning")

# For training
parser.add_argument("--ealy_stopping_patience", type=int, default=3, help="the patience of early stopping")
parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
parser.add_argument("--total_steps", type=int, default=500000, help="Number of steps to train")
parser.add_argument("--val_check_interval", type=int, default=10)
parser.add_argument("--num_train_data", type=int, default=-1,
                    help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
parser.add_argument("--num_val_data", type=int, default=-1, help="The number of testing data")
parser.add_argument("--accum_data_per_step", type=int, default=16, help="Number of data per step")
parser.add_argument("--label_smoothing_factor", type=float, default=0.1, help="Label smoothing")
parser.add_argument("--optimizer", type=str, default="adafactor")
parser.add_argument("--lr_scheduler_type", type=str, default="linear")

# For testing
parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
parser.add_argument("--length_penalty", type=float, default=1, help="length penalty of generated text")
parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                    help="The size of no repeat ngram in generation")
parser.add_argument("--num_test_data", type=int, default=-1, help="The number of testing data")

args = parser.parse_args()
print(args)

wandb.login()
project_name = "led_large_16384_%s" % args.dataset_name
wandb.init(project=project_name)

# load dataset
dataset_train = load_dataset('json', data_files=args.data_path + '%s_train.jsonl' % args.dataset_name, split='all')
if args.num_train_data > 0:
    dataset_train = dataset_train.shuffle(seed=args.rand_seed).select(range(args.num_train_data))
print("dataset train", len(dataset_train))

dataset_val = load_dataset('json', data_files=args.data_path + '%s_valid.jsonl' % args.dataset_name, split='all')
if args.num_val_data > 0:
    dataset_val = dataset_val.shuffle(seed=args.rand_seed).select(range(args.num_val_data))
print("dataset dev", len(dataset_val))

dataset_test = load_dataset('json', data_files=args.data_path + '%s_test.jsonl' % args.dataset_name, split='all')
if args.num_test_data > 0:
    dataset_test = dataset_test.shuffle(seed=args.rand_seed).select(range(args.num_test_data))
print("dataset test", len(dataset_test))

# load tokenizer
tokenizer = LEDTokenizer.from_pretrained(args.pretrained_model)
config = LEDConfig.from_pretrained(args.pretrained_model)
print(config)
config.gradient_checkpointing = args.gradient_checkpointing
# set generate hyper-parameters
config.num_beams = args.beam_size
config.max_length = args.max_length_tgt
config.min_length = args.min_length_tgt
config.length_penalty = args.length_penalty
config.early_stopping = True
config.no_repeat_ngram_size = args.no_repeat_ngram_size

# load model + enable gradient checkpointing & disable cache for checkpointing
led = LEDForConditionalGeneration.from_pretrained(args.pretrained_model, config=config)

# training parameters
max_input_length = args.max_length_input
max_output_length = args.max_length_tgt
batch_size = args.batch_size


def process_data_to_model_inputs(batch):
    documents = []
    for source_documents in batch["source_documents"]:
        max_length_doc = max_input_length // len(source_documents)
        input_text = []
        for source_document in source_documents:
            length = 0
            all_sents = sent_tokenize(source_document)
            for s in all_sents:
                input_text.append(s)
                length += len(s.split())
                if length >= max_length_doc:
                    break
        documents.append(" ".join(input_text))

    summaries = batch["meta_review"]

    # tokenize the inputs and labels
    input_dict = tokenizer(documents, padding='max_length', max_length=max_input_length,
                           truncation=True)
    outputs = tokenizer(
        summaries,
        padding="max_length",
        truncation=True,
        max_length=max_output_length
        )

    results = {}
    results["input_ids"] = input_dict.input_ids
    results["attention_mask"] = input_dict.attention_mask
    attention_mask = input_dict.attention_mask
    # create 0 global_attention_mask lists
    results["global_attention_mask"] = len(results["input_ids"]) * [
        [0 for _ in range(len(results["input_ids"][0]))]
        ]
    # since above lists are references, the following line changes the 0 index for all samples
    results["global_attention_mask"][0][0] = 1

    labels = outputs.input_ids
    # We have to make sure that the PAD token is ignored
    results["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in ls]
        for ls in labels
        ]

    results["decoder_input_ids"] = led.prepare_decoder_input_ids_from_labels(torch.tensor(results["labels"]))

    return results


print("Preprocessing dataset train")
dataset_train = dataset_train.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
    )

print("Preprocessing dataset validation")
dataset_val = dataset_val.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
    )

print("Preprocessing dataset test")
dataset_test = dataset_test.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
    )

# set Python list to PyTorch tensor
dataset_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "decoder_input_ids"],
    )

# set Python list to PyTorch tensor
dataset_val.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "decoder_input_ids"],
    )

# set Python list to PyTorch tensor
dataset_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "decoder_input_ids"],
    )

training_args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    do_predict=True,
    predict_with_generate=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="%s/checkpoints" % args.save_path,
    logging_strategy="steps",
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=args.val_check_interval,
    save_strategy="steps",
    save_steps=args.val_check_interval,
    load_best_model_at_end=True,
    warmup_steps=args.warmup_steps,
    save_total_limit=args.save_top_k,
    gradient_accumulation_steps=args.accum_data_per_step,
    max_steps=args.total_steps,
    label_smoothing_factor=args.label_smoothing_factor,
    learning_rate=args.lr,
    report_to='wandb',
    optim=args.optimizer,
    lr_scheduler_type=args.lr_scheduler_type
    )


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.ealy_stopping_patience)]
    )

# test_results = trainer.predict(test_dataset=dataset_test)
# print(test_results.metrics)
# start training
trainer.train()
test_results = trainer.predict(test_dataset=dataset_test)
print(test_results.metrics)
