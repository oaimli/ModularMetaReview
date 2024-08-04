from transformers import LlamaTokenizer, AutoModelForCausalLM

# model_name="meta-llama/Llama-2-7b-chat-hf"
# model_name="meta-llama/Llama-2-7b-hf"
model_name="/scratch/punim0521/model_weights/llama-meta/weights-hf/7B"

tokenizer = LlamaTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True,
    )
model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    )
print(model.config)

text = "I am a student in Melbourne."
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
print(tokenizer(
            [text],
            return_tensors="pt",
            padding="do_not_pad",
            max_length=4,
            truncation=True
        ))