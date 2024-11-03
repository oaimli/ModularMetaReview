from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
import numpy as np
import json
from alignscore import AlignScore


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
    scorer = AlignScore(model='roberta-large', batch_size=8, device='cuda:0',
                        ckpt_path='../../tmp/AlignScore/AlignScore-large.ckpt',
                        evaluation_mode='nli_sp')

    model_names = ["gpt_4o", "llama31_70b", "llama31_8b", "mixtral8x7b_v01"]
    facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
    for model_name in model_names:
        with open(f"scientific_reasoning_result_{model_name}.json") as f:
            samples = json.load(f)

        sources = []
        ground_truths = []
        generations = []
        pair_facets = []
        for key in samples.keys():
            sample = samples[key]
            categorization_pairs = sample["categorization_pairs"]
            for pair in categorization_pairs:
                review_fargments = pair["review_fragments"]
                meta_review_fragments = pair["meta_review_fragments"]
                if len(review_fargments) > 0 and len(meta_review_fragments) > 0:
                    sources.append(" ".join(review_fargments))
                    ground_truths.append(" ".join(meta_review_fragments))
                    generations.append(pair["meta_generated"])
                    pair_facets.append(pair["facet"])

        scores = rouge_corpus(ground_truths, generations, types=['rouge1', 'rouge2', 'rougeLsum'])

        print("Average ROUGE",
              (scores["rouge1"]["fmeasure"] + scores["rouge2"]["fmeasure"] + scores["rougeLsum"]["fmeasure"]) / 3)

        scores_align = scorer.score(contexts=sources, claims=generations)
        score_avg = np.mean(scores_align)
        print("AlignScore", score_avg)

        for facet in facets:
            tmp = []
            for score, pair_facet in zip(scores_align, pair_facets):
                if pair_facet == facet:
                    tmp.append(score)
            print(f"AlignScore in {facet}", np.mean(tmp), len(tmp))






