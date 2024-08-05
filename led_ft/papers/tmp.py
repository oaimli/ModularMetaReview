from datasets import load_dataset

dataset_all = load_dataset('json', data_files='../../datasets/peermeta_train.jsonl', split='all')
print("dataset all", len(dataset_all))