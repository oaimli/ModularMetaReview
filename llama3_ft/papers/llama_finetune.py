from transformers import HfArgumentParser, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import EarlyStoppingCallback
from arguments import ModelArguments, DataArguments, TrainingArguments
from dataloader import get_data_module
import wandb
from torch.distributed import get_rank, is_initialized


def train(model_args, data_args, training_args):
    all_args = {}
    all_args.update(vars(data_args))
    all_args.update(vars(model_args))
    all_args.update(vars(training_args))

    if is_initialized():
        if get_rank() == 0:
            wandb.init(project=training_args.project_name, config=all_args, name=training_args.run_name)
    else:
        wandb.init(project=training_args.project_name, config=all_args, name=training_args.run_name)

    # load model and tokenizer
    if training_args.resume_from_checkpoint != None:
        model_in_use = training_args.resume_from_checkpoint
    else:
        model_in_use = model_args.model_name_or_path
    print("loading model:", model_in_use)
    tokenizer_class = LlamaTokenizer if "llama" in model_args.model_name_or_path else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_in_use, padding_side="right",
                                                model_max_length=model_args.max_length_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_in_use, trust_remote_code=True)

    # there is only no pad token for LLaMA-2 and "/scratch/punim0521/model_weights/llama-meta/weights-hf/7B"
    print("bos", tokenizer.bos_token, tokenizer.bos_token_id)
    print("eos", tokenizer.eos_token, tokenizer.eos_token_id)
    print("pad", tokenizer.pad_token, tokenizer.pad_token_id)
    print("unk", tokenizer.unk_token, tokenizer.unk_token_id)
    print("model_max_length", tokenizer.model_max_length)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<pad-token>"
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<unk-token>"
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    print(model.config)


    # data_module = get_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=data_module["train_dataset"],
    #                   eval_dataset=data_module["eval_dataset"], data_collator=data_module["data_collator"],
    #                   callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    data_module = get_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=data_module["train_dataset"],
                      eval_dataset=data_module["eval_dataset"], data_collator=data_module["data_collator"])

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train:
        train(model_args, data_args, training_args)
