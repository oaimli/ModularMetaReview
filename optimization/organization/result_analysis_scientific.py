import datetime
import json
import time

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
        source_data = annotation_data[id]  # the original annotation data

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


def transform(fragment, whole_text, vector):
    # whole_text_words = whole_text.split()
    start = 0
    while start >= 0:
        start = whole_text.find(fragment, start)
        if start != -1:
            prefix = whole_text[:start].split()
            target = fragment.split()
            # print(len(whole_text.split()))
            # print(len(prefix))
            # print(target)
            # print(whole_text_words[len(prefix)], target[0])
            # assert whole_text_words[len(prefix)] == target[0]

            for i in range(len(prefix), len(prefix) + len(target)):
                if i < len(vector):
                    vector[i] = 1

            start += len(fragment)


def words_sharing(vector_a, vector_b):
    assert len(vector_a) == len(vector_b)
    sharing_count = 0
    for item_a, item_b in zip(vector_a, vector_b):
        if item_a == item_b == 1:
            sharing_count += 1
    ones_a = 0
    for item in vector_a:
        if item == 1:
            ones_a += 1
    ones_b = 0
    for item in vector_b:
        if item == 1:
            ones_b += 1
    r = (sharing_count + 1) / (ones_a + 1)
    p = (sharing_count + 1) / (ones_b + 1)
    return r, p, 2*(r*p)/(r+p)


def word_level_agreement(results_1, results_2, annotation_data):
    novelty_agreements_fmeasure = []
    novelty_agreements_recall = []
    novelty_agreements_precision = []
    soundness_agreements_fmeasure = []
    soundness_agreements_recall = []
    soundness_agreements_precision = []
    clarity_agreements_fmeasure = []
    clarity_agreements_recall = []
    clarity_agreements_precision = []
    advancement_agreements_fmeasure = []
    advancement_agreements_recall = []
    advancement_agreements_precision = []
    compliance_agreements_fmeasure = []
    compliance_agreements_recall = []
    compliance_agreements_precision = []
    overall_agreements_fmeasure = []
    overall_agreements_recall = []
    overall_agreements_precision = []
    meta_review_agreements_fmeasure = []
    meta_review_agreements_recall = []
    meta_review_agreements_precision = []
    review_agreements_fmeasure = []
    review_agreements_recall = []
    review_agreements_precision = []
    all_text_agreements_fmeasure = []
    all_text_agreements_recall = []
    all_text_agreements_precision = []

    for id in annotation_data.keys():
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

        print(id)
        result_1 = results_1[id]  # the annotation result of the first annotator
        result_2 = results_2[id]  # the annotation result of the second annotator
        source_data = annotation_data[id]  # the original annotation data

        meta_review = source_data["meta_review"]
        meta_review_categorization_1 = result_1["meta_review_categorization"]
        meta_review_categorization_2 = result_2["meta_review_categorization"]

        words_meta_review = meta_review.split()

        meta_review_signal_1 = [0] * len(words_meta_review)
        meta_review_signal_2 = [0] * len(words_meta_review)
        meta_review_signal_1_novelty = [0] * len(words_meta_review)
        meta_review_signal_2_novelty = [0] * len(words_meta_review)
        meta_review_signal_1_soundness = [0] * len(words_meta_review)
        meta_review_signal_2_soundness = [0] * len(words_meta_review)
        meta_review_signal_1_clarity = [0] * len(words_meta_review)
        meta_review_signal_2_clarity = [0] * len(words_meta_review)
        meta_review_signal_1_advancement = [0] * len(words_meta_review)
        meta_review_signal_2_advancement = [0] * len(words_meta_review)
        meta_review_signal_1_compliance = [0] * len(words_meta_review)
        meta_review_signal_2_compliance = [0] * len(words_meta_review)
        meta_review_signal_1_overall = [0] * len(words_meta_review)
        meta_review_signal_2_overall = [0] * len(words_meta_review)

        for facet, fragments in meta_review_categorization_1.items():
            for fragment in fragments:
                transform(fragment, meta_review, meta_review_signal_1)

                if facet == "Novelty":
                    transform(fragment, meta_review, meta_review_signal_1_novelty)

                if facet == "Soundness":
                    transform(fragment, meta_review, meta_review_signal_1_soundness)

                if facet == "Clarity":
                    transform(fragment, meta_review, meta_review_signal_1_clarity)

                if facet == "Advancement":
                    transform(fragment, meta_review, meta_review_signal_1_advancement)

                if facet == "Compliance":
                    transform(fragment, meta_review, meta_review_signal_1_compliance)

                if facet == "Overall":
                    transform(fragment, meta_review, meta_review_signal_1_overall)
        meta_reviews_1s.extend(meta_review_signal_1)
        facets_1s_novelty.extend(meta_review_signal_1_novelty)
        facets_1s_soundness.extend(meta_review_signal_1_soundness)
        facets_1s_clarity.extend(meta_review_signal_1_clarity)
        facets_1s_advancement.extend(meta_review_signal_1_advancement)
        facets_1s_compliance.extend(meta_review_signal_1_compliance)
        facets_1s_overall.extend(meta_review_signal_1_overall)

        for facet, fragments in meta_review_categorization_2.items():
            for fragment in fragments:
                transform(fragment, meta_review, meta_review_signal_2)

                if facet == "Novelty":
                    transform(fragment, meta_review, meta_review_signal_2_novelty)

                if facet == "Soundness":
                    transform(fragment, meta_review, meta_review_signal_2_soundness)

                if facet == "Clarity":
                    transform(fragment, meta_review, meta_review_signal_2_clarity)

                if facet == "Advancement":
                    transform(fragment, meta_review, meta_review_signal_2_advancement)

                if facet == "Compliance":
                    transform(fragment, meta_review, meta_review_signal_2_compliance)

                if facet == "Overall":
                    transform(fragment, meta_review, meta_review_signal_2_overall)
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
            words_review_content = review_content.split()

            review_signal_1 = [0] * len(words_review_content)
            review_signal_2 = [0] * len(words_review_content)
            review_signal_1_novelty = [0] * len(words_review_content)
            review_signal_2_novelty = [0] * len(words_review_content)
            review_signal_1_soundness = [0] * len(words_review_content)
            review_signal_2_soundness = [0] * len(words_review_content)
            review_signal_1_clarity = [0] * len(words_review_content)
            review_signal_2_clarity = [0] * len(words_review_content)
            review_signal_1_advancement = [0] * len(words_review_content)
            review_signal_2_advancement = [0] * len(words_review_content)
            review_signal_1_compliance = [0] * len(words_review_content)
            review_signal_2_compliance = [0] * len(words_review_content)
            review_signal_1_overall = [0] * len(words_review_content)
            review_signal_2_overall = [0] * len(words_review_content)

            for facet, fragments in review_categorization_1.items():
                for fragment in fragments:
                    transform(fragment, review_content, review_signal_1)

                    if facet == "Novelty":
                        transform(fragment, review_content, review_signal_1_novelty)

                    if facet == "Soundness":
                        transform(fragment, review_content, review_signal_1_soundness)

                    if facet == "Clarity":
                        transform(fragment, review_content, review_signal_1_clarity)

                    if facet == "Advancement":
                        transform(fragment, review_content, review_signal_1_advancement)

                    if facet == "Compliance":
                        transform(fragment, review_content, review_signal_1_compliance)

                    if facet == "Overall":
                        transform(fragment, review_content, review_signal_1_overall)

            for facet, fragments in review_categorization_2.items():
                for fragment in fragments:
                    transform(fragment, review_content, review_signal_2)

                    if facet == "Novelty":
                        transform(fragment, review_content, review_signal_2_novelty)

                    if facet == "Soundness":
                        transform(fragment, review_content, review_signal_2_soundness)

                    if facet == "Clarity":
                        transform(fragment, review_content, review_signal_2_clarity)

                    if facet == "Advancement":
                        transform(fragment, review_content, review_signal_2_advancement)

                    if facet == "Compliance":
                        transform(fragment, review_content, review_signal_2_compliance)

                    if facet == "Overall":
                        transform(fragment, review_content, review_signal_2_overall)

            reviews_1s.extend(review_signal_1)
            facets_1s_novelty.extend(review_signal_1_novelty)
            facets_1s_soundness.extend(review_signal_1_soundness)
            facets_1s_clarity.extend(review_signal_1_clarity)
            facets_1s_advancement.extend(review_signal_1_advancement)
            facets_1s_compliance.extend(review_signal_1_compliance)
            facets_1s_overall.extend(review_signal_1_overall)

            reviews_2s.extend(review_signal_2)
            facets_2s_novelty.extend(review_signal_2_novelty)
            facets_2s_soundness.extend(review_signal_2_soundness)
            facets_2s_clarity.extend(review_signal_2_clarity)
            facets_2s_advancement.extend(review_signal_2_advancement)
            facets_2s_compliance.extend(review_signal_2_compliance)
            facets_2s_overall.extend(review_signal_2_overall)

            print(len(review_signal_1), len(review_signal_2))

        a = np.array(meta_reviews_1s + reviews_1s)
        b = np.array(meta_reviews_2s + reviews_2s)
        print(len(meta_reviews_1s), len(meta_reviews_2s), len(reviews_1s), len(reviews_2s))
        r, p, f = words_sharing(a, b)
        all_text_agreements_fmeasure.append(f)
        all_text_agreements_recall.append(r)
        all_text_agreements_precision.append(p)

        a = np.array(meta_reviews_1s)
        b = np.array(meta_reviews_2s)
        r, p, f = words_sharing(a, b)
        meta_review_agreements_fmeasure.append(f)
        meta_review_agreements_recall.append(r)
        meta_review_agreements_precision.append(p)

        a = np.array(reviews_1s)
        b = np.array(reviews_2s)
        r, p, f = words_sharing(a, b)
        review_agreements_fmeasure.append(f)
        review_agreements_recall.append(r)
        review_agreements_precision.append(p)

        a = np.array(facets_1s_novelty)
        b = np.array(facets_2s_novelty)
        r, p, f = words_sharing(a, b)
        novelty_agreements_fmeasure.append(f)
        novelty_agreements_recall.append(r)
        novelty_agreements_precision.append(p)

        a = np.array(facets_1s_soundness)
        b = np.array(facets_2s_soundness)
        r, p, f = words_sharing(a, b)
        soundness_agreements_fmeasure.append(f)
        soundness_agreements_recall.append(r)
        soundness_agreements_precision.append(p)

        a = np.array(facets_1s_clarity)
        b = np.array(facets_2s_clarity)
        r, p, f = words_sharing(a, b)
        clarity_agreements_fmeasure.append(f)
        clarity_agreements_recall.append(r)
        clarity_agreements_precision.append(p)

        a = np.array(facets_1s_advancement)
        b = np.array(facets_2s_advancement)
        r, p, f = words_sharing(a, b)
        advancement_agreements_fmeasure.append(f)
        advancement_agreements_recall.append(r)
        advancement_agreements_precision.append(p)

        a = np.array(facets_1s_compliance)
        b = np.array(facets_2s_compliance)
        r, p, f = words_sharing(a, b)
        compliance_agreements_fmeasure.append(f)
        compliance_agreements_recall.append(r)
        compliance_agreements_precision.append(p)

        a = np.array(facets_1s_overall)
        b = np.array(facets_2s_overall)
        r, p, f = words_sharing(a, b)
        overall_agreements_fmeasure.append(f)
        overall_agreements_recall.append(r)
        overall_agreements_precision.append(p)

    result = {}
    result["Highlight correlation, meta-review + review, word level (r)"] = np.mean(all_text_agreements_recall)
    result["Highlight correlation, meta-review + review, word level (p)"] = np.mean(all_text_agreements_precision)
    result["Highlight correlation, meta-review + review, word level (f1)"] = np.mean(all_text_agreements_fmeasure)
    result["Highlight correlation, meta-review, word level (r)"] = np.mean(meta_review_agreements_recall)
    result["Highlight correlation, meta-review, word level (p)"] = np.mean(meta_review_agreements_precision)
    result["Highlight correlation, meta-review, word level (f1)"] = np.mean(meta_review_agreements_fmeasure)
    result["Highlight correlation, review, word level (r)"] = np.mean(review_agreements_recall)
    result["Highlight correlation, review, word level (p)"] = np.mean(review_agreements_precision)
    result["Highlight correlation, review, word level (f1)"] = np.mean(review_agreements_fmeasure)
    result["Highlight correlation, meta-review + review, word level, novelty (r)"] = np.mean(novelty_agreements_recall)
    result["Highlight correlation, meta-review + review, word level, novelty (p)"] = np.mean(novelty_agreements_precision)
    result["Highlight correlation, meta-review + review, word level, novelty (f1)"] = np.mean(novelty_agreements_fmeasure)
    result["Highlight correlation, meta-review + review, word level, soundness (r)"] = np.mean(soundness_agreements_recall)
    result["Highlight correlation, meta-review + review, word level, soundness (p)"] = np.mean(soundness_agreements_precision)
    result["Highlight correlation, meta-review + review, word level, soundness (f1)"] = np.mean(soundness_agreements_fmeasure)
    result["Highlight correlation, meta-review + review, word level, clarity (r)"] = np.mean(clarity_agreements_recall)
    result["Highlight correlation, meta-review + review, word level, clarity (p)"] = np.mean(clarity_agreements_precision)
    result["Highlight correlation, meta-review + review, word level, clarity (f1)"] = np.mean(clarity_agreements_fmeasure)
    result["Highlight correlation, meta-review + review, word level, advancement (r)"] = np.mean(advancement_agreements_recall)
    result["Highlight correlation, meta-review + review, word level, advancement (p)"] = np.mean(advancement_agreements_precision)
    result["Highlight correlation, meta-review + review, word level, advancement (f1)"] = np.mean(advancement_agreements_fmeasure)
    result["Highlight correlation, meta-review + review, word level, compliance (r)"] = np.mean(compliance_agreements_recall)
    result["Highlight correlation, meta-review + review, word level, compliance (p)"] = np.mean(compliance_agreements_precision)
    result["Highlight correlation, meta-review + review, word level, compliance (f1)"] = np.mean(compliance_agreements_fmeasure)
    result["Highlight correlation, meta-review + review, word level, overall (r)"] = np.mean(overall_agreements_recall)
    result["Highlight correlation, meta-review + review, word level, overall (p)"] = np.mean(overall_agreements_precision)
    result["Highlight correlation, meta-review + review, word level, overall (f1)"] = np.mean(overall_agreements_fmeasure)

    return result


if __name__ == "__main__":
    model_name = "gpt-4o-2024-05-13"
    with open("scientific_categorization_result_gpt_4o_processed.json") as f:
        model_results = json.load(f)

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
    assert annotation_data_share.keys() == bryan_results_share.keys() == zenan_results_share.keys() == model_results_share.keys()

    print("################ Agreement Bryan and Zenan: ################")
    result_bz = word_level_agreement(bryan_results_share, zenan_results_share, annotation_data_share)

    print("################ Agreement Bryan and Model: ################")
    result_bm = word_level_agreement(bryan_results_share, model_results_share, annotation_data_share)

    print("################ Agreement Zenan and Model: ################")
    result_zm = word_level_agreement(zenan_results_share, model_results_share, annotation_data_share)

    print(shared_ids)

    print(f"################ All agreements, A1<->A2, A1<->Model and A2<->Model {model_name} {datetime.datetime.now()} ################")
    for key in result_bm:
        print(key, "------", result_bz[key], result_bm[key], result_zm[key])


    model_name = "Mixtral-8x7B-Instruct-v0.1"
    with open("scientific_categorization_result_mixtral8x7b_v01_processed.json") as f:
        model_results = json.load(f)

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
    assert annotation_data_share.keys() == bryan_results_share.keys() == zenan_results_share.keys() == model_results_share.keys()

    print("################ Agreement Bryan and Zenan: ################")
    result_bz = word_level_agreement(bryan_results_share, zenan_results_share, annotation_data_share)

    print("################ Agreement Bryan and Model: ################")
    result_bm = word_level_agreement(bryan_results_share, model_results_share, annotation_data_share)

    print("################ Agreement Zenan and Model: ################")
    result_zm = word_level_agreement(zenan_results_share, model_results_share, annotation_data_share)

    print(shared_ids)

    print(
        f"################ All agreements, A1<->A2, A1<->Model and A2<->Model {model_name} {datetime.datetime.now()} ################")
    for key in result_bm:
        print(key, "------", result_bz[key], result_bm[key], result_zm[key])


    model_name = "LLaMA3.1-70B-Instruct"
    with open("scientific_categorization_result_llama31_70b_processed.json") as f:
        model_results = json.load(f)

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
    assert annotation_data_share.keys() == bryan_results_share.keys() == zenan_results_share.keys() == model_results_share.keys()

    print("################ Agreement Bryan and Zenan: ################")
    result_bz = word_level_agreement(bryan_results_share, zenan_results_share, annotation_data_share)

    print("################ Agreement Bryan and Model: ################")
    result_bm = word_level_agreement(bryan_results_share, model_results_share, annotation_data_share)

    print("################ Agreement Zenan and Model: ################")
    result_zm = word_level_agreement(zenan_results_share, model_results_share, annotation_data_share)

    print(shared_ids)

    print(
        f"################ All agreements, A1<->A2, A1<->Model and A2<->Model {model_name} {datetime.datetime.now()} ################")
    for key in result_bm:
        print(key, "------", result_bz[key], result_bm[key], result_zm[key])
