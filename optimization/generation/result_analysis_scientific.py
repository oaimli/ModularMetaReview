import json
import numpy as np

with open("scientific_generation_result_gpt4.json") as f:
    samples = json.load(f)

sources = []
ground_truths = []
generations = []
for key in samples.keys():
    sample = samples[key]

    reviews = []
    for r in sample["reviews"]:
        reviews.append(r["comment"])
    sources.append("\n".join(reviews))
    ground_truths.append(sample["meta_review"])
    generations.append(sample["meta_review_generated"])

from rouge_score import rouge_scorer
from nltk import sent_tokenize
def rouge(reference, candidate, types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True,
          split_summaries=True):
    """
    This is based on rouge-score 0.0.4
    If using rougeLsum, it is necessary to split sentences with '\n' in summaries in advance
    """
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

def rouge_corpus(references, candidates, types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True,
                 split_summaries=True):
    if len(references) != len(candidates):
        print("len must be equal")
        return None
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

rouge_results = rouge_corpus(ground_truths, generations)
print("ROUGE-1 F1", rouge_results["rouge1"]["fmeasure"], "ROUGE-2 F1", rouge_results["rouge2"]["fmeasure"], "ROUGE-L F1", rouge_results["rougeLsum"]["fmeasure"],)

# import sys
# sys.path.append("../../")
# from utils.metrics import evaluating_summaries_single_source
# print(evaluating_summaries_single_source(ground_truths, generations, sources))
#
#
# from utils.metrics import summac_scores
# scores_zs, scores_conv = summac_scores(sources, generations)
# print("scores zs generation", np.mean(scores_zs), "scores conv generation", np.mean(scores_conv))
# scores_zs, scores_conv = summac_scores(sources, ground_truths)
# print("scores zs ground truths", np.mean(scores_zs), "scores conv ground truth", np.mean(scores_conv))