# Post-process the extraction results of models
# to make sure that extracted fragments are exactly the same as texts in the original document
import json


def matching_fragments(document, facet_fragments):
    # trace judgements back to the original document, and return the corresponding fragments

    sentences = []
    for sent in nlp(document).sents:
        sentences.append(sent.text)
    # sentences = sent_tokenize(document)

    result = {}
    review_facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
    for facet in review_facets:
        fragments = set([])
        for judgement in judgements:
            judgement_facet = judgement["Criteria Facet"]
            if facet == judgement_facet:
                content_expression = judgement["Content Expression"]
                sentiment_expression = judgement["Sentiment Expression"]
                # anchored fragment in the original document
                target = content_expression + " " + sentiment_expression
                rouges = []
                for sentence in sentences:
                    scores = scorer.score(target, sentence)
                    rouges.append(scores["rouge2"].fmeasure + scores["rouge1"].fmeasure + scores["rougeLsum"].fmeasure)
                fragments.add(sentences[rouges.index(max(rouges))])

        result[facet] = list(fragments)
    return result


if __name__ == "__main__":
    file_name = "scientific_categorization_result_gpt4"

    with open(f"{file_name}_gpt4.json") as f:
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
        samples = json.dump(samples, f)
