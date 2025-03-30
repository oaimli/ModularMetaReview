from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
import numpy as np
import json


def rouge(reference, candidate, types=None, use_stemmer=True, split_summaries=True):
    """
    This is based on rouge-score 0.0.4
    If using rougeLsum, it is necessary to split sentences with '\n' in summaries in advance
    """
    if types == None:
        types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

    if 'rougeLsum' in types and split_summaries:
        reference = '\n'.join(sent_tokenize(reference))
        candidate = '\n'.join(sent_tokenize(candidate))

    results = {}
    for t in types:
        if t not in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            print("The type must be selected in rouge1, rouge2, rougeL, and rougeLsum.")
            return results
    scorer = rouge_scorer.RougeScorer(types, use_stemmer=use_stemmer)
    scores = scorer.score(reference, candidate)
    for t in types:
        r = {}
        r["precision"] = scores[t].precision
        r["recall"] = scores[t].recall
        r["fmeasure"] = scores[t].fmeasure
        results[t] = r
    return results


def rouge_corpus(references, candidates, types=None, use_stemmer=True, split_summaries=True):
    assert len(references) == len(candidates)

    if types == None:
        types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

    results = {}
    for t in types:
        s = {}
        s['recall'] = []
        s['precision'] = []
        s['fmeasure'] = []
        results[t] = s
    for ref, can in zip(references, candidates):
        s = rouge(ref, can, types=types, use_stemmer=use_stemmer, split_summaries=split_summaries)
        for t in types:
            results[t]['recall'].append(s[t]['recall'])
            results[t]['precision'].append(s[t]['precision'])
            results[t]['fmeasure'].append(s[t]['fmeasure'])

    final_results = {}
    for t in types:
        s = results[t]
        tmp = {}
        tmp['precision'] = np.mean(s['precision'])
        tmp['recall'] = np.mean(s['recall'])
        tmp['fmeasure'] = np.mean(s['fmeasure'])
        final_results[t] = tmp
    return final_results


if __name__ == "__main__":
    with open("all.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]

    # calculate ROUGE based on human references
    print("########### ROUGE based on references ################")
    for dataset_name in dataset_names:
        print(dataset_name)
        generations_info = info[dataset_name]
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            with open(generation_file) as f:
                samples = json.load(f)

            candidates = []
            references = []
            for sample in samples:
                if "comment" not in sample.keys():
                    candidates.append(sample[candidate_key])
                    if isinstance(sample[reference_key], str):
                        references.append(sample[reference_key])
                    else:
                        references.append(sample[reference_key][0]) # SPACE has multiple references

            scores = rouge_corpus(references, candidates, types=['rouge1', 'rouge2', 'rougeLsum'])

            print("Average F1",
                  (scores["rouge1"]["fmeasure"] + scores["rouge2"]["fmeasure"] + scores["rougeLsum"]["fmeasure"]) / 3)

    # calculate ROUGE based on source reviews
    print("########### ROUGE based on source reviews ################")
    for dataset_name in dataset_names:
        print(dataset_name)
        generations_info = info[dataset_name]
        reference_calculation = -1

        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            with open(generation_file) as f:
                samples = json.load(f)

            if reference_calculation == -1:
                print("human reference")
                references = []
                source_texts = []
                for sample in samples:
                    if "comment" not in sample.keys():
                        if isinstance(sample[reference_key], str):
                            references.append(sample[reference_key])
                        else:
                            references.append(sample[reference_key][0])  # SPACE has multiple references
                        source_texts.append("\n".join(sample["source_documents"]))

                scores = rouge_corpus(source_texts, references, types=['rouge1', 'rouge2', 'rougeLsum'])

                print("Average F1",
                      (scores["rouge1"]["fmeasure"] + scores["rouge2"]["fmeasure"] + scores["rougeLsum"][
                          "fmeasure"]) / 3)
                reference_calculation = 0

            candidates = []
            source_texts = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                source_texts.append("\n".join(sample["source_documents"]))

            scores = rouge_corpus(source_texts, candidates, types=['rouge1', 'rouge2', 'rougeLsum'])

            print("Average F1",
                  (scores["rouge1"]["fmeasure"] + scores["rouge2"]["fmeasure"] + scores["rougeLsum"][
                      "fmeasure"]) / 3)

    # calculate lengths of generations
    print("########### lengths of generated meta-reviews ################")
    for dataset_name in dataset_names:
        print(dataset_name)
        generations_info = info[dataset_name]
        reference_calculation = -1
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            with open(generation_file) as f:
                samples = json.load(f)

            if reference_calculation == -1:
                print("human references")
                reference_lengths = []
                for sample in samples:
                    if "comment" not in sample.keys():
                        if isinstance(sample[reference_key], str):
                            reference = sample[reference_key]
                        else:
                            reference = sample[reference_key][0]  # SPACE has multiple references
                        reference_lengths.append(len(reference.split()))

                print("Average length of generations:", np.mean(reference_lengths))
                reference_calculation = 0

            candidate_lengths = []
            for sample in samples:
                candidate_lengths.append(len(sample[candidate_key].split()))

            print("Average length of generations:", np.mean(candidate_lengths))

