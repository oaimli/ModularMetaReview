# import jsonlines
# import json
#
# results = []
# with jsonlines.open("/home/miao4/punim0521/ModularMetaReview/results/llama31_8b_amasum_shoes/generated_summaries.jsonl") as reader:
#     for line in reader:
#         results.append(line)
#
# with open("/home/miao4/punim0521/ModularMetaReview/results/llama31_8b_amasum_shoes/generated_summaries.json", "w") as f:
#     json.dump(results, f)


import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="flash_attention_2")
print(model.config)