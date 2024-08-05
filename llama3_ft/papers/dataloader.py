import copy
import random
import logging
from datasets import load_dataset
from typing import Dict, Sequence
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

IGNORE_INDEX = -100


class DialogueDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict, tokenizer: PreTrainedTokenizer, model_args):
        super(DialogueDataset, self).__init__()

        self.tokenizer = tokenizer
        self.model_args = model_args

        self.all_samples = []
        for sample in list_data_dict:
            conversation_history = " ".join(sample["conversation_history"][-10:])
            knowledge_source = sample["knowledge_source"]
            output_text = sample["response"]

            output_text_tokenized = self.tokenizer.encode(
                output_text,
                max_length=self.model_args.max_predict_length,
                truncation=True
            )

            conversation_tokenized = self.tokenizer.encode(conversation_history,
                                                           max_length=self.tokenizer.model_max_length - len(
                                                               output_text_tokenized),
                                                           truncation=True)
            knowledge_max_length = self.tokenizer.model_max_length - len(output_text_tokenized) - len(
                conversation_tokenized) - 6
            knowledge_source_tokenized = self.tokenizer.encode(knowledge_source,
                                                               max_length=knowledge_max_length if knowledge_max_length > 0 else 0,
                                                               truncation=True)
            # print(len(output_text_tokenized), len(conversation_tokenized), len(knowledge_source_tokenized))
            # sample_input_ids = torch.tensor(
            #     knowledge_source_tokenized + conversation_tokenized + output_text_tokenized + [self.tokenizer.eos_token_id])
            sample_text = self.tokenizer.decode(knowledge_source_tokenized,
                                                skip_special_tokens=True) + self.tokenizer.decode(
                conversation_tokenized,
                skip_special_tokens=False) + self.tokenizer.decode(
                output_text_tokenized, skip_special_tokens=False) + " " + self.tokenizer.eos_token
            sample_text_dict = self.tokenizer(sample_text, return_tensors="pt", padding="do_not_pad", truncation=False)
            sample_input_ids = sample_text_dict.input_ids[0]
            if sample_input_ids[-1] != 2:
                print("the text has been truncated")
            # print(sample_input_ids)
            # do not need to shift to left as it is calculated in loss calculation process in llama model
            sample_labels = copy.deepcopy(sample_input_ids)
            sample_labels[:-(len(output_text_tokenized) + 1)] = IGNORE_INDEX
            # print("sample_input_ids", len(sample_input_ids), sample_input_ids)
            # print("sample_labels", len(sample_labels), sample_labels)
            self.all_samples.append({"sample_input_ids": sample_input_ids, "sample_labels": sample_labels})


    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.all_samples[i]
        sample_input_ids = sample["sample_input_ids"]
        sample_labels = sample["sample_labels"]
        return dict(input_ids=sample_input_ids, labels=sample_labels)


@dataclass
class DialogueDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # print(input_ids.shape, labels.shape, input_ids.ne(self.tokenizer.pad_token_id).shape)
        # print(input_ids)
        # print(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )


def get_data_module(tokenizer: PreTrainedTokenizer, data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.info("Loading data...")
    all_data = load_dataset('json', data_files=data_args.dataset_path + '/%s.jsonl' % data_args.dataset_name,
                            split='all')

    training_data = all_data.filter(lambda s: s['label'] == 'train')
    if data_args.num_training_samples > 0:
        training_data = training_data.select(
            random.choices(range(len(training_data)), k=data_args.num_training_samples))
    if data_args.keep_split > 0:
        # in this case num_train_data needs to be -1
        all_count = len(training_data)
        split_count = int(all_count / 5)
        all_indexes = range(all_count)
        selected_indexes = []
        for i in all_indexes:
            if i < (data_args.keep_split - 1) * split_count or (
                    i >= data_args.keep_split * split_count and i < all_count):
                selected_indexes.append(i)
        training_data = training_data.select(selected_indexes)
    logging.info("all training data", len(training_data))

    evaluation_data = all_data.filter(lambda s: s['label'] == 'val')
    if data_args.num_val_samples > 0:
        evaluation_data = evaluation_data.select(
            random.choices(range(len(evaluation_data)), k=data_args.num_val_samples))
    logging.info("all evaluation data", len(evaluation_data))

    test_data = all_data.filter(lambda s: s['label'] == 'test')
    if data_args.num_test_samples > 0:
        test_data = test_data.select(random.choices(range(len(test_data)), k=data_args.num_test_samples))
    logging.info("all test data", len(test_data))

    logging.info("Formatting and tokenizing training data")
    train_dataset = DialogueDataset(tokenizer=tokenizer, list_data_dict=training_data, model_args=model_args)
    # print("train", train_dataset[0])
    logging.info("Formatting and tokenizing evaluation data")
    eval_dataset = DialogueDataset(tokenizer=tokenizer, list_data_dict=evaluation_data, model_args=model_args)
    # print("evaluation", eval_dataset[0])
    logging.info("Formatting and tokenizing test data")
    test_dataset = DialogueDataset(tokenizer=tokenizer, list_data_dict=test_data, model_args=model_args)
    data_collator = DialogueDataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, test_dataset=test_dataset,
                data_collator=data_collator)
