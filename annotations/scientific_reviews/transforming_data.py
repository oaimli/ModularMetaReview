# To make it easy to evaluate the categorization results
# We have to transform the annotated data into the same format as our automatic extracted fragments

import json
import spacy
from rouge_score import rouge_scorer
from nltk import sent_tokenize

def matching_fragments(document, judgements):
    # trace judgements back to the original document, and return the corresponding fragments

    # sentences = nlp(document).sents
    sentences = sent_tokenize(document)

    result = {}
    review_facets = ["Novelty", "Soundness", "Clarity", "Advancement", "Compliance", "Overall"]
    for facet in review_facets:
        fragments = []
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
                fragments.append(sentences[rouges.index(max(rouges))])

        result[facet] = fragments
    return result



def transform_annotations(samples, annotations):
    transformed_annotations = {}
    for sample_key, sample_value in samples.items():
        print(sample_key)
        # the sample_value is a dictionary
        annotation_sample = annotations[sample_key] # the annotation_sample is a list

        # meta-review
        meta_review = sample_value["meta_review"]
        meta_review_title = sample_value["meta_review_title"]
        meta_review_annotation = {}
        for annotation in annotation_sample:
            document_title = annotation["Document Title"]
            if meta_review_title == document_title:
                meta_review_annotation = annotation
                break
        assert len(meta_review_annotation.keys()) > 0
        facet_fragments = matching_fragments(meta_review, meta_review_annotation["Annotated Judgements"])
        sample_value["meta_review_categorization"] = facet_fragments

        # reviews
        reviews = sample_value["reviews"]
        review_categorization = []
        for review in reviews:
            review_comment = review["comment"]
            review_annotation = {}
            review_title = review["title"]
            for annotation in annotation_sample:
                document_title = annotation["Document Title"]
                if review_title == document_title:
                    review_annotation = annotation
                    break
            assert len(review_annotation.keys()) > 0
            facet_fragments = matching_fragments(review_comment, meta_review_annotation["Annotated Judgements"])
            review_categorization.append(facet_fragments)
        sample_value["review_categorization"] = review_categorization

        transformed_annotations[sample_key] = sample_value

    return transformed_annotations


if __name__ == "__main__":
    # nlp = spacy.load("en_core_web_sm")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    with open("annotation_data_small.json") as f:
        samples = json.load(f)

    with open("br_annotation_result.json") as f:
        br_annotations = json.load(f)
    br_transformed = transform_annotations(samples, br_annotations)
    with open("br_annotation_result_fragments.json", "w") as f:
        json.dump(br_transformed, f, indent=4)

    with open("ze_annotation_result.json") as f:
        ze_annotations = json.load(f)
    ze_transformed = transform_annotations(samples, ze_annotations)
    with open("ze_annotation_result_fragments.json", "w") as f:
        json.dump(ze_transformed, f, indent=4)
