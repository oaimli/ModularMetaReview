import random
import torch
from datasets import load_dataset
from transformers import LEDTokenizer, LEDForConditionalGeneration
from nltk.tokenize import sent_tokenize
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="The path to save model checkpoints, logs and results")
parser.add_argument("--data_path", type=str, default="../../datasets/")
parser.add_argument("--dataset_name", type=str, default="peermeta")
parser.add_argument("--num_test_data", type=int, default=512)
parser.add_argument("--max_length_source", type=int, default=1024)
parser.add_argument("--max_length_tgt", type=int, default=512)
parser.add_argument("--min_length_tgt", type=int, default=0)
parser.add_argument("--pretrained_model", type=str, default="led-large-16384")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_beams", type=int, default=5)
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
parser.add_argument("--length_penalty", type=float, default=0.6)
parser.add_argument("--rand_seed", type=int, default=42)
args = parser.parse_args()
print(args)

# load data
dataset_test = load_dataset('json', data_files=args.data_path + '%s_test.jsonl' % args.dataset_name, split='all')
print("dataset test", len(dataset_test))
if len(dataset_test) > args.num_test_data > 0:
    dataset_test = dataset_test.select(random.choices(range(len(dataset_test)), k=args.num_test_data))
print("dataset test selected", len(dataset_test))

# load tokenizer
tokenizer = LEDTokenizer.from_pretrained(args.pretrained_model)
model = LEDForConditionalGeneration.from_pretrained(args.pretrained_model).to("cuda")


def batch_process(batch):
    documents = []
    for source_documents in batch["source_documents"]:
        max_length_doc = args.max_length_source // len(source_documents)

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

    input_dict = tokenizer(documents, padding="max_length", max_length=args.max_length_source, return_tensors="pt",
                           truncation=True)
    input_ids = input_dict.input_ids.to("cuda")

    global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_abstract_ids = model.generate(input_ids=input_ids, global_attention_mask=global_attention_mask,
                                            max_length=args.max_length_tgt, min_length=args.min_length_tgt,
                                            num_beams=args.num_beams, do_sample=args.do_sample, top_p=args.top_p,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                                            length_penalty=args.length_penalty)

    result = {}
    result["source_documents"] = batch["source_documents"]
    result['meta_review'] = batch["meta_review"]
    result["generated_meta_review"] = tokenizer.batch_decode(predicted_abstract_ids.tolist(), skip_special_tokens=True)
    result["label"] = batch["label"]
    return result


results = dataset_test.map(batch_process, batched=True, batch_size=args.batch_size)
print("the length of results", len(results))
print("the length of results summary", len(results["summary"]))

# print generations
data_idx = random.choices(range(len(results)), k=2)
for item in results.select(data_idx):
    print("#### Meta-review: ", item['meta_review'])
    print("#### Generated meta-review:", item['generated_meta_review'])

# save generations
with open(f"{args.save_path}/generations_led_large.json") as f:
    json.dump(results, f)

