# Post-process the extraction results of models
# to make sure that extracted fragments are exactly the same as texts in the original document
import json
import spacy
from rouge_score import rouge_scorer


def matching_fragments(document, facet_fragments):
    # trace judgements back to the original document, and return the corresponding fragments

    sentences = []
    for sent in nlp(document).sents:
        sentences.append(sent.text)

    result = {}
    for facet, fragments in facet_fragments.items():
        fragments_new = set([])
        for fragment in fragments:
            if fragment in document:
                if len(fragment) > 10:
                    fragments_new.add(fragment)
            else:
                fragment_sentences = []
                for sent in nlp(fragment).sents:
                    fragment_sentences.append(sent.text)
                for fragment_sentence in fragment_sentences:
                    if len(fragment_sentence) > 10:
                        if fragment_sentence in document:
                            fragments_new.add(fragment_sentence)
                        else:
                            rouges = []
                            for sentence in sentences:
                                scores = scorer.score(fragment_sentence, sentence)
                                rouges.append(scores["rouge2"].fmeasure + scores["rouge1"].fmeasure + scores[
                                    "rougeLsum"].fmeasure)
                            fragments_new.add(sentences[rouges.index(max(rouges))])
        result[facet] = list(fragments_new)

    return result


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    file_name = "scientific_categorization_result_gpt_4o"
    # file_name = "scientific_categorization_result_llama31_70b"
    # file_name = "scientific_categorization_result_mixtral8x7b_v01"

    with open(f"{file_name}.json") as f:
        samples = json.load(f)

    for sample_key, sample_value in samples.items():
        meta_review = sample_value["meta_review"]
        meta_review_categorization = sample_value["meta_review_categorization"]
        samples[sample_key]["meta_review_categorization"] = matching_fragments(meta_review, meta_review_categorization)

        reviews = sample_value["reviews"]
        review_categorization = sample_value["review_categorization"]
        review_categorization_new = []
        for review, categorization in zip(reviews, review_categorization):
            review_categorization_new.append(matching_fragments(review["comment"], categorization))
        samples[sample_key]["review_categorization"] = review_categorization_new

    with open(f"{file_name}_processed.json", "w") as f:
        samples = json.dump(samples, f, indent=4)
