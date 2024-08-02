import json
from sklearn.metrics import cohen_kappa_score
import krippendorff
import numpy as np
from scipy import stats


def krippendorffs_alpha(a, b):
    # nominal kripppendorff's alpha
    return krippendorff.alpha(reliability_data=np.array([a, b]), level_of_measurement="nominal")


def character_level_agreement(results_1, results_2, annotation_data):
    facets_1s_novelty = []
    facets_2s_novelty = []
    facets_1s_soundness = []
    facets_2s_soundness = []
    facets_1s_clarity = []
    facets_2s_clarity = []
    facets_1s_advancement = []
    facets_2s_advancement = []
    facets_1s_compliance = []
    facets_2s_compliance = []
    facets_1s_overall = []
    facets_2s_overall = []
    meta_reviews_1s = []
    meta_reviews_2s = []
    reviews_1s = []
    reviews_2s = []

    for id in annotation_data.keys():
        print(id)
        result_1 = results_1[id]  # the annotation result of the first annotator
        result_2 = results_2[id]  # the annotation result of the second annotator
        source_data = annotation_data[key]  # the original annotation data

        meta_review = source_data["meta_review"]
        meta_review_categorization_1 = result_1["meta_review_categorization"]
        meta_review_categorization_2 = result_2["meta_review_categorization"]

        meta_review_signal_1 = [0] * len(meta_review)
        meta_review_signal_2 = [0] * len(meta_review)
        meta_review_signal_1_novelty = [0] * len(meta_review)
        meta_review_signal_2_novelty = [0] * len(meta_review)
        meta_review_signal_1_soundness = [0] * len(meta_review)
        meta_review_signal_2_soundness = [0] * len(meta_review)
        meta_review_signal_1_clarity = [0] * len(meta_review)
        meta_review_signal_2_clarity = [0] * len(meta_review)
        meta_review_signal_1_advancement = [0] * len(meta_review)
        meta_review_signal_2_advancement = [0] * len(meta_review)
        meta_review_signal_1_compliance = [0] * len(meta_review)
        meta_review_signal_2_compliance = [0] * len(meta_review)
        meta_review_signal_1_overall = [0] * len(meta_review)
        meta_review_signal_2_overall = [0] * len(meta_review)

        for facet, fragments in meta_review_categorization_1.items():
            for fragment in fragments:
                start = 0
                while start >= 0:
                    start = meta_review.find(fragment, start)
                    if start != -1:
                        meta_review_signal_1[start: start + len(fragment)] = [1] * len(fragment)
                        start += len(fragment)

            if facet == "Novelty":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_1_novelty[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Soundness":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_1_soundness[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Clarity":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_1_clarity[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Advancement":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_1_advancement[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Compliance":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_1_compliance[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Overall":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_1_overall[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)
        meta_reviews_1s.extend(meta_review_signal_1)
        facets_1s_novelty.extend(meta_review_signal_1_novelty)
        facets_1s_soundness.extend(meta_review_signal_1_soundness)
        facets_1s_clarity.extend(meta_review_signal_1_clarity)
        facets_1s_advancement.extend(meta_review_signal_1_advancement)
        facets_1s_compliance.extend(meta_review_signal_1_compliance)
        facets_1s_overall.extend(meta_review_signal_1_overall)

        for facet, fragments in meta_review_categorization_2.items():
            for fragment in fragments:
                start = 0
                while start >= 0:
                    start = meta_review.find(fragment, start)
                    if start != -1:
                        meta_review_signal_2[start: start + len(fragment)] = [1] * len(fragment)
                        start += len(fragment)

            if facet == "Novelty":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_2_novelty[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Soundness":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_2_soundness[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Clarity":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_2_clarity[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Advancement":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_2_advancement[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Compliance":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_2_compliance[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

            if facet == "Overall":
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = meta_review.find(fragment, start)
                        if start != -1:
                            meta_review_signal_2_overall[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)
        meta_reviews_2s.extend(meta_review_signal_2)
        facets_2s_novelty.extend(meta_review_signal_2_novelty)
        facets_2s_soundness.extend(meta_review_signal_2_soundness)
        facets_2s_clarity.extend(meta_review_signal_2_clarity)
        facets_2s_advancement.extend(meta_review_signal_2_advancement)
        facets_2s_compliance.extend(meta_review_signal_2_compliance)
        facets_2s_overall.extend(meta_review_signal_2_overall)

        reviews = source_data["reviews"]
        review_categorizations_1 = result_1["review_categorization"]
        review_categorizations_2 = result_2["review_categorization"]
        for review, review_categorization_1, review_categorization_2 in zip(reviews, review_categorizations_1,
                                                                            review_categorizations_2):
            review_content = review["comment"]
            review_signal_1 = [0] * len(review_content)
            review_signal_2 = [0] * len(review_content)
            review_signal_1_novelty = [0] * len(review_content)
            review_signal_2_novelty = [0] * len(review_content)
            review_signal_1_soundness = [0] * len(review_content)
            review_signal_2_soundness = [0] * len(review_content)
            review_signal_1_clarity = [0] * len(review_content)
            review_signal_2_clarity = [0] * len(review_content)
            review_signal_1_advancement = [0] * len(review_content)
            review_signal_2_advancement = [0] * len(review_content)
            review_signal_1_compliance = [0] * len(review_content)
            review_signal_2_compliance = [0] * len(review_content)
            review_signal_1_overall = [0] * len(review_content)
            review_signal_2_overall = [0] * len(review_content)

            for facet, fragments in review_categorization_1.items():
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = review_content.find(fragment, start)
                        if start != -1:
                            review_signal_1[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

                if facet == "Novelty":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_1_novelty[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Soundness":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_1_soundness[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Clarity":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_1_clarity[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Advancement":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_1_advancement[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Compliance":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_1_compliance[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Overall":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_1_overall[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)
            reviews_1s.extend(review_signal_1)
            facets_1s_novelty.extend(review_signal_1_novelty)
            facets_1s_soundness.extend(review_signal_1_soundness)
            facets_1s_clarity.extend(review_signal_1_clarity)
            facets_1s_advancement.extend(review_signal_1_advancement)
            facets_1s_compliance.extend(review_signal_1_compliance)
            facets_1s_overall.extend(review_signal_1_overall)

            for facet, fragments in review_categorization_2.items():
                for fragment in fragments:
                    start = 0
                    while start >= 0:
                        start = review_content.find(fragment, start)
                        if start != -1:
                            review_signal_2[start: start + len(fragment)] = [1] * len(fragment)
                            start += len(fragment)

                if facet == "Novelty":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_2_novelty[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Soundness":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_2_soundness[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Clarity":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_2_clarity[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Advancement":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_2_advancement[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Compliance":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_2_compliance[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)

                if facet == "Overall":
                    for fragment in fragments:
                        start = 0
                        while start >= 0:
                            start = review_content.find(fragment, start)
                            if start != -1:
                                review_signal_2_overall[start: start + len(fragment)] = [1] * len(fragment)
                                start += len(fragment)
            reviews_2s.extend(review_signal_2)
            facets_2s_novelty.extend(review_signal_2_novelty)
            facets_2s_soundness.extend(review_signal_2_soundness)
            facets_2s_clarity.extend(review_signal_2_clarity)
            facets_2s_advancement.extend(review_signal_2_advancement)
            facets_2s_compliance.extend(review_signal_2_compliance)
            facets_2s_overall.extend(review_signal_2_overall)

    result = {}
    print("#### Highlight correlation, meta-review+review, character level")
    a = np.array(meta_reviews_1s + reviews_1s)
    b = np.array(meta_reviews_2s + reviews_2s)
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
    result["Highlight correlation, meta-review + review, character level, Cohen Kappa"] = cohen_kappa_score_result
    result["Highlight correlation, meta-review + review, character level, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review + review, character level, Kendall Tall"] = kendalltau_result[0]
    result["Highlight correlation, meta-review + review, character level, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review + review, character level, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, meta-review, character level")
    a = np.array(meta_reviews_1s)
    b = np.array(meta_reviews_2s)
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
    result["Highlight correlation, meta-review, character level, Cohen Kappa"] = cohen_kappa_score_result
    result["Highlight correlation, meta-review, character level, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review, character level, Kendall Tall"] = kendalltau_result[0]
    result["Highlight correlation, meta-review, character level, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review, character level, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, review, character level")
    a = np.array(reviews_1s)
    b = np.array(reviews_2s)
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
    result["Highlight correlation, review, character level, Cohen Kappa"] = cohen_kappa_score_result
    result["Highlight correlation, review, character level, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, review, character level, Kendall Tall"] = kendalltau_result[0]
    result["Highlight correlation, review, character level, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, review, character level, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, meta-review + review, character level, novelty")
    a = np.array(facets_1s_novelty)
    b = np.array(facets_2s_novelty)
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
    result[
        "Highlight correlation, meta-review + review, character level, novelty, Cohen Kappa"] = cohen_kappa_score_result
    result[
        "Highlight correlation, meta-review + review, character level, novelty, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review + review, character level, novelty, Kendall Tall"] = kendalltau_result[0]
    result["Highlight correlation, meta-review + review, character level, novelty, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review + review, character level, novelty, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, meta-review + review, character level, soundness")
    a = np.array(facets_1s_soundness)
    b = np.array(facets_2s_soundness)
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
    result[
        "Highlight correlation, meta-review + review, character level, soundness, Cohen Kappa"] = cohen_kappa_score_result
    result[
        "Highlight correlation, meta-review + review, character level, soundness, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review + review, character level, soundness, Kendall Tall"] = kendalltau_result[
        0]
    result["Highlight correlation, meta-review + review, character level, soundness, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review + review, character level, soundness, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, meta-review + review, character level, clarity")
    a = np.array(facets_1s_clarity)
    b = np.array(facets_2s_clarity)
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
    result[
        "Highlight correlation, meta-review + review, character level, clarity, Cohen Kappa"] = cohen_kappa_score_result
    result[
        "Highlight correlation, meta-review + review, character level, clarity, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review + review, character level, clarity, Kendall Tall"] = kendalltau_result[
        0]
    result["Highlight correlation, meta-review + review, character level, clarity, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review + review, character level, clarity, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, meta-review + review, character level, advancement")
    a = np.array(facets_1s_advancement)
    b = np.array(facets_2s_advancement)
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
    result[
        "Highlight correlation, meta-review + review, character level, advancement, Cohen Kappa"] = cohen_kappa_score_result
    result[
        "Highlight correlation, meta-review + review, character level, advancement, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review + review, character level, advancement, Kendall Tall"] = \
        kendalltau_result[
            0]
    result["Highlight correlation, meta-review + review, character level, advancement, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review + review, character level, advancement, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, meta-review + review, character level, compliance")
    a = np.array(facets_1s_compliance)
    b = np.array(facets_2s_compliance)
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
    result[
        "Highlight correlation, meta-review + review, character level, compliance, Cohen Kappa"] = cohen_kappa_score_result
    result[
        "Highlight correlation, meta-review + review, character level, compliance, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review + review, character level, compliance, Kendall Tall"] = \
        kendalltau_result[
            0]
    result["Highlight correlation, meta-review + review, character level, compliance, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review + review, character level, compliance, Pearson"] = pearsonr_result[0]

    print("#### Highlight correlation, meta-review + review, character level, overall")
    a = np.array(facets_1s_overall)
    b = np.array(facets_2s_overall)
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
    result[
        "Highlight correlation, meta-review + review, character level, overall, Cohen Kappa"] = cohen_kappa_score_result
    result[
        "Highlight correlation, meta-review + review, character level, overall, Krippendorff Alpha"] = krippendorff_alpha
    result["Highlight correlation, meta-review + review, character level, overall, Kendall Tall"] = \
        kendalltau_result[
            0]
    result["Highlight correlation, meta-review + review, character level, overall, Spearman"] = spearmanr_result[0]
    result["Highlight correlation, meta-review + review, character level, overall, Pearson"] = pearsonr_result[0]

    return result


if __name__ == "__main__":
    # with open("scientific_categorization_result_gpt4_processed.json") as f:
    #     model_results = json.load(f)
    with open("scientific_categorization_result_llama3_70b_processed.json") as f:
        model_results = json.load(f)
    # with open("scientific_categorization_result_llama3_1_70b_processed.json") as f:
    #     model_results = json.load(f)

    with open("../../annotations/scientific_reviews/br_annotation_result_fragments.json") as f:
        bryan_results = json.load(f)
    with open("../../annotations/scientific_reviews/ze_annotation_result_fragments.json") as f:
        zenan_results = json.load(f)
    with open("../../annotations/scientific_reviews/annotation_data_small.json") as f:
        annotation_data = json.load(f)

    bryan_results_share = {}
    zenan_results_share = {}
    model_results_share = {}
    annotation_data_share = {}
    shared_ids = list(
        set(bryan_results.keys()).intersection(set(zenan_results.keys())).intersection(set(model_results.keys())))
    for key in shared_ids:
        bryan_results_share[key] = bryan_results[key]
        zenan_results_share[key] = zenan_results[key]
        model_results_share[key] = model_results[key]
        annotation_data_share[key] = annotation_data[key]

    print("Br", len(bryan_results_share), "Ze", len(zenan_results_share), "Model", len(model_results_share),
          "Annotation data", len(annotation_data_share))
    assert len(annotation_data_share.keys()) == len(bryan_results_share.keys()) == len(
        zenan_results_share.keys())

    print("################ Agreement Bryan and Zenan: ################")
    result_bz = character_level_agreement(bryan_results_share, zenan_results_share, annotation_data_share)

    print("################ Agreement Bryan and Model: ################")
    result_bm = character_level_agreement(bryan_results_share, model_results_share, annotation_data_share)

    print("################ Agreement Zenan and Model: ################")
    result_zm = character_level_agreement(zenan_results_share, model_results_share, annotation_data_share)

    print("################ Overall agreement, A1<->A2, A1<->Model and A2<->Model ################")
    for key in result_bm:
        print(key, "------", result_bz[key], result_bm[key], result_zm[key])
