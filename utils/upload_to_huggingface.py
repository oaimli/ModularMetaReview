# Test the model and upload them to the huggingface hub
from transformers import LlamaTokenizer, LlamaForCausalLM
import huggingface_hub

huggingface_hub.login(token="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")

PATH_TO_CONVERTED_WEIGHTS = "/home/miao4/punim0521/model_weights/llama-meta/weights-hf/13B"
PATH_TO_CONVERTED_TOKENIZER = "/home/miao4/punim0521/model_weights/llama-meta/weights-hf/13B"

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to("cuda")
tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

model.push_to_hub("oaimli/llama-13b")
tokenizer.push_to_hub("oaimli/llama-13b")

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])