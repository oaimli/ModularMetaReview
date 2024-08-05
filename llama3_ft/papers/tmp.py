from transformers import AutoModelForCausalLM, AutoTokenizer


model = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model, padding_side="right",
                                                model_max_length=17408, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

print("bos", tokenizer.bos_token, tokenizer.bos_token_id)
print("eos", tokenizer.eos_token, tokenizer.eos_token_id)
print("pad", tokenizer.pad_token, tokenizer.pad_token_id)
print("unk", tokenizer.unk_token, tokenizer.unk_token_id)
print("model_max_length", tokenizer.model_max_length)