import json
import numpy as np
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize


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
        r["precision"] = '%.4f' % scores[t].precision
        r["recall"] = '%.4f' % scores[t].recall
        r["fmeasure"] = '%.4f' % scores[t].fmeasure
        results[t] = r
    return results


def analyzing(targets, sources):
    # lengths = []
    # for target in targets:
    #     lengths.append(len(target.split()))
    # print("Mean", np.mean(lengths), "Max", np.max(lengths), "Min", np.min(lengths), "Std", np.std(lengths))

    rouges = []
    for target, source in zip(targets, sources):
        scores = rouge(target, source)
        rouges.append((scores["rouge1"]["fmeasure"] + scores["rouge2"]["fmeasure"] + scores["rougeLsum"]["fmeasure"]) / 3)
    print("Mean", np.mean(rouges), "Max", np.max(rouges), "Min", np.min(rouges), "Std", np.std(rouges))






if __name__ == "__main__":
    with open("../eval_auto/info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)

        facets = []
        if dataset_name == "peermeta":
            facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
        if dataset_name == "space":
            facets = ["Building", "Cleanliness", "Food", "Location", "Rooms", "Service"]
        if dataset_name == "amasum_shoes":
            facets = ["Breathability", "Durability", "Weight", "Cushioning", "Stability", "Flexibility", "Traction",
                      "Sizefit", "Comfort", "Misc"]

        generations_info = info[dataset_name]

        # all generation results share the same review-categorization
        generation_file_target = ""
        for generation_info in generations_info:
            if generation_info["model_name"] == "modular_llama3":
                generation_file_target = generation_info["generation_file"]
        categorization_file = "_".join(generation_file_target.split("/")[1:]).split(".")[0] + ".json"
        with open("../eval_auto/categorization/" + categorization_file) as f:
            samples = json.load(f)
        source_texts = []
        for sample in samples:
            source_texts.append("\n".join(sample["source_documents"]))


        # human reference
        generation_file = generations_info[0]["generation_file"]
        print("human reference")
        candidate_key = generations_info[0]["candidate_key"]
        reference_key = generations_info[0]["reference_key"]

        categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
        with open("../eval_auto/categorization/" + categorization_file) as f:
            samples = json.load(f)

        references = []
        reference_categorizations = []
        for sample in samples:
            if isinstance(sample[reference_key], str):
                references.append(sample[reference_key])
            else:
                references.append(sample[reference_key][0])  # SPACE has multiple references
            reference_categorizations.append(sample["categorization_reference"])
        analyzing(references, source_texts)

        # model generations
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]

            # use the processed result with shared content from categorization
            categorization_file = "_".join(generation_file.split("/")[1:]).split(".")[0] + ".json"
            with open("../eval_auto/categorization/" + categorization_file) as f:
                samples = json.load(f)

            candidates = []
            candidate_categorizations = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                candidate_categorizations.append(sample["categorization_candidate"])

            analyzing(candidates, source_texts)