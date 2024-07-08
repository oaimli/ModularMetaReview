import json


def character_level_agreement(human_results_1, model_results_2, annotation_data):
    content_1s = []
    content_2s = []
    sentiment_1s = []
    sentiment_2s = []
    all_1s = []
    all_2s = []

    for id, source_documents in annotation_data.items():
        print(id)
        result_1 = human_results_1[id]
        result_2 = model_results_2[id]

        for source_document_dict in source_documents:
            title = source_document_dict["title"]
            original_document = source_document_dict["content"]

            judgements_1_tmp = []
            for document in result_1:
                if title == document["Document Title"]:
                    judgements_1_tmp = document["Annotated Judgements"]
                    break
            if len(judgements_1_tmp) == 0:
                print("No judgements in result-1", title)

            signal_content_1 = [0] * len(original_document)
            signal_sentiment_1 = [0] * len(original_document)
            signal_all_1 = [0] * len(original_document)
            for judgement in judgements_1_tmp:
                content = judgement["Content Expression"]
                sentiment = judgement["Sentiment Expression"]
                start = 0
                while start >= 0:
                    start = original_document.find(content, start)
                    if start != -1:
                        signal_content_1[start: start + len(content)] = [1] * len(content)
                        signal_all_1[start: start + len(content)] = [1] * len(content)
                        start += len(content)
                start = 0
                while start >= 0:
                    start = original_document.find(sentiment, start)
                    if start != -1:
                        signal_sentiment_1[start: start + len(sentiment)] = [1] * len(sentiment)
                        signal_all_1[start: start + len(sentiment)] = [1] * len(sentiment)
                        start += len(sentiment)
            content_1s.extend(signal_content_1)
            sentiment_1s.extend(signal_sentiment_1)
            all_1s.extend(signal_all_1)

            extracted_fragments = result_2["review_categorization"]


            signal_content_2 = [0] * len(original_document)
            signal_sentiment_2 = [0] * len(original_document)
            signal_all_2 = [0] * len(original_document)
            for judgement in judgements_2_tmp:
                content = judgement["Content Expression"]
                sentiment = judgement["Sentiment Expression"]
                start = 0
                while start >= 0:
                    start = original_document.find(content, start)
                    if start != -1:
                        signal_content_2[start: start + len(content)] = [1] * len(content)
                        signal_all_2[start: start + len(content)] = [1] * len(content)
                        start += len(content)
                start = 0
                while start >= 0:
                    start = original_document.find(sentiment, start)
                    if start != -1:
                        signal_sentiment_2[start: start + len(sentiment)] = [1] * len(sentiment)
                        signal_all_2[start: start + len(sentiment)] = [1] * len(sentiment)
                        start += len(sentiment)
            content_2s.extend(signal_content_2)
            sentiment_2s.extend(signal_sentiment_2)
            all_2s.extend(signal_all_2)

    result = {}
    print("#### Highlight correlation, content, character level")
    a = np.array(content_1s)
    b = np.array(content_2s)
    cohen_kappa_score_result = cohen_kappa_score(a, b)
    krippendorff_alpha = krippendorffs_alpha(a, b)
    kendalltau_result = stats.kendalltau(a, b)
    spearmanr_result = stats.spearmanr(a, b)
    pearsonr_result = stats.pearsonr(a, b)
    print("Cohen Kappa: ", cohen_kappa_score_result)
    print("Krippendorff Alpha", krippendorff_alpha)
    print("Kendall Tau: ", kendalltau_result)
    print("Spearman: ", spearmanr_result)
    print("Pearson: ", pearsonr_result)
    result["Highlight correlation, content, character level, Cohen Kappa"] = cohen_kappa_score_result
    result["Highlight correlation, content, character level, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, content, character level, Kendall Tall"] = kendalltau_result[0]
    result["Highlight correlation, content, character level, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, content, character level, Pearson"] = pearsonr_result[0]


    print("#### Highlight correlation, sentiment, character level")
    a = np.array(sentiment_1s)
    b = np.array(sentiment_2s)
    cohen_kappa_score_result = cohen_kappa_score(a, b)
    krippendorff_alpha = krippendorffs_alpha(a, b)
    kendalltau_result = stats.kendalltau(a, b)
    spearmanr_result = stats.spearmanr(a, b)
    pearsonr_result = stats.pearsonr(a, b)
    print("Cohen Kappa: ", cohen_kappa_score_result)
    print("Krippendorff Alpha", krippendorff_alpha)
    print("Kendall Tau: ", kendalltau_result)
    print("Spearman: ", spearmanr_result)
    print("Pearson: ", pearsonr_result)
    result["Highlight correlation, sentiment, character level, Cohen Kappa"] = cohen_kappa_score_result
    result["Highlight correlation, sentiment, character level, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, sentiment, character level, Kendall Tall"] = kendalltau_result[0]
    result["Highlight correlation, sentiment, character level, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, sentiment, character level, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, content+sentiment, character level")
    a = np.array(all_1s)
    b = np.array(all_2s)
    cohen_kappa_score_result = cohen_kappa_score(a, b)
    krippendorff_alpha = krippendorffs_alpha(a, b)
    kendalltau_result = stats.kendalltau(a, b)
    spearmanr_result = stats.spearmanr(a, b)
    pearsonr_result = stats.pearsonr(a, b)
    print("Cohen Kappa: ", cohen_kappa_score_result)
    print("Krippendorff Alpha", krippendorff_alpha)
    print("Kendall Tau: ", kendalltau_result)
    print("Spearman: ", spearmanr_result)
    print("Pearson: ", pearsonr_result)
    result["Highlight correlation, content+sentiment, character level, Cohen Kappa"] = cohen_kappa_score_result
    result["Highlight correlation, content+sentiment, character level, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, content+sentiment, character level, Kendall Tall"] = kendalltau_result[0]
    result["Highlight correlation, content+sentiment, character level, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, content+sentiment, character level, Pearson"] = pearsonr_result[0]

    return result


if __name__ == "__main__":
    with open("../../annotations/scientific_reviews/br_annotation_result.json") as f:
        bryan_results = json.load(f)
    with open("../../annotations/scientific_reviews/ze_annotation_result.json") as f:
        zenan_results = json.load(f)
    with open("scientific_categorization_result_gpt4.json") as f:
        gpt4_results = json.load(f)
    with open("../../annotations/scientific_reviews/annotation_data_small.json") as f:
        annotation_data = json.load(f)

    bryan_results_share = {}
    zenan_results_share = {}
    gpt4_results_share = {}
    annotation_data_share = {}
    shared_ids = list(
        set(bryan_results.keys()).intersection(set(zenan_results.keys())).intersection(set(gpt4_results.keys())))
    for key in shared_ids:
        bryan_results_share[key] = bryan_results[key]
        zenan_results_share[key] = zenan_results[key]
        gpt4_results_share[key] = gpt4_results[key]
        annotation_data_share[key] = annotation_data[key]

    print("Br", len(bryan_results_share), "Ze", len(zenan_results_share), "GPT-4", len(gpt4_results_share),
          "Annotation data", len(annotation_data_share))

    for key in shared_ids:
        bryan_result = bryan_results_share[key]
        zenan_result = zenan_results_share[key]
        gpt4_result = gpt4_results_share[key]
        source_data = annotation_data_share[key]

        source_data_new = []
        source_data_new.append(
            {"title": source_data["meta_review_title"], "content": source_data["meta_review"]})
        for review in source_data["reviews"]:
            source_data_new.append(
                {"title": review["title"], "content": review["comment"]})
        annotation_data_share[key] = source_data_new

        bryan_results_share[key] = bryan_result
        zenan_results_share[key] = zenan_result
        gpt4_results_share[key] = gpt4_result

    assert len(annotation_data.keys()) == len(bryan_results_share.keys()) == len(zenan_results_share.keys())

    print("################ Annotator Agreement Bryan and GPT-4: ################")
    result_bg = character_level_agreement(bryan_results_share, gpt4_results_share, annotation_data_share)

    print("################ Annotator Agreement Zenan and GPT-4: ################")
    result_zg = character_level_agreement(zenan_results_share, gpt4_results_share, annotation_data_share)

    print(
        "################ Overall results for agreement, A1<->A2, A1<->GPT-4 and A2<->GPT-4: ################")
    for key in result_bg:
        print(key, "------", result_bg[key], result_zg[key])

