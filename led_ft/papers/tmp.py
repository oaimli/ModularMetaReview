from datasets import load_dataset

dataset_all = load_dataset('json', data_files='../../peermeta/data/peermeta_all.json', split='all')
print("dataset all", len(dataset_all))