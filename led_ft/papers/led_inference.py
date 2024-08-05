import random
import torch
from datasets import load_dataset
from transformers import LEDTokenizer, LEDForConditionalGeneration
from nltk.tokenize import sent_tokenize
import sys
import argparse
import os
import json

sys.path.append('../../../')
from utils.metrics import evaluating_summaries_multi_sources

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="The path to save model checkpoints, logs and results")
parser.add_argument("--data_path", type=str, default="../../datasets/")
parser.add_argument("--dataset_name", type=str, default="multinews")
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
parser.add_argument("--rand_seed", type=int, default=42,
                    help="Seed for random sampling, useful for few shot learning")
args = parser.parse_args()
print(args)

# load data
dataset_all = load_dataset('json', data_files=args.data_path + '%s.json' % args.dataset_name, split='all')
print("dataset all", len(dataset_all))

# random.seed(42)  # This is to control random selection
dataset_test = dataset_all.filter(lambda s: s['label'] == 'test')
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

        # documents.append(" ".join(source_documents))

    input_dict = tokenizer(documents, padding="max_length", max_length=args.max_length_source, return_tensors="pt",
                           truncation=True)
    input_ids = input_dict.input_ids.to("cuda")

    # # This can also be done separately
    # input_ids = []
    # for document in documents:
    #     input_ids.append(torch.tensor(tokenizer.encode(document, max_length=max_length_source, truncation=True)))
    # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
    #                                             padding_value=tokenizer.pad_token_id).to("cuda")

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
    result['summary'] = batch["summary"]
    result["predicted_summary"] = tokenizer.batch_decode(predicted_abstract_ids.tolist(), skip_special_tokens=True)
    return result


results = dataset_test.map(batch_process, batched=True, batch_size=args.batch_size)
print("the length of results", len(results))
print("the length of results summary", len(results["summary"]))

# scores = rouge_corpus(references=results["summary"], candidates=results["predicted_summary"],
#                       types=["rouge1", "rouge2", "rougeL", "rougeLsum"], split_summaries=True)
# print("LED inference result on %s:"%args.dataset_name)
# print("rouge-1", scores["rouge1"])
# print("rouge-2", scores["rouge2"])
# print("rouge-L", scores["rougeL"])
# print("rouge-Lsum", scores["rougeLsum"])

# print generated summaries
data_idx = random.choices(range(len(results)), k=2)
for item in results.select(data_idx):
    print("####Abstract: ", item['summary'])
    print("####Predicted_summary:", item['predicted_summary'])

# save generated summaries
output_dir = "%s/generated_summaries" % args.save_path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
print("Exited results in the folder before saving", len(os.listdir(output_dir)))
for item in results:
    idx = len(os.listdir(output_dir))
    result_dict = {}
    result_dict["prediction"] = item['predicted_summary']
    result_dict["reference"] = item['summary']
    result_dict["source_documents"] = item["source_documents"]
    with open(os.path.join(output_dir, "%d.json" % (idx)), "w") as f:
        json.dump(result_dict, f)
print("Exited results in the folder after saving", len(os.listdir(output_dir)))

predictions = []
references = []
source_document_clusters = []
for item in results:
    predictions.append(item['predicted_summary'])
    references.append(item['summary'])
    source_document_clusters.append(item["source_documents"])
print(evaluating_summaries_multi_sources(source_document_clusters=source_document_clusters, gold_summaries=references,
                                         generated_summaries=predictions))
