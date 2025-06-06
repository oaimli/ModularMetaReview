# Transforming crawled raw data into peermeta
# Removing reviews which are written after the meta-review
import json
import time
import random
import jsonlines
import numpy as np


def split_data(papers):
    all_num = len(papers)
    train_num = int(all_num * 0.8)
    val_num = int(all_num * 0.1)

    papers_indexes = range(all_num)
    papers_train_indexes = random.sample(papers_indexes, train_num)
    for i in papers_train_indexes:
        papers[i]["label"] = "train"
    print("train", len(papers_train_indexes))

    papers_val_test_indexes = [item for item in papers_indexes if item not in papers_train_indexes]
    papers_val_indexes = random.sample(papers_val_test_indexes, val_num)
    for i in papers_val_indexes:
        papers[i]["label"] = "val"
    print("val", len(papers_val_indexes))

    papers_test_indexes = [item for item in papers_val_test_indexes if item not in papers_val_indexes]
    for i in papers_test_indexes:
        papers[i]["label"] = "test"
    print("test", len(papers_test_indexes))


def prepare_iclr(year):
    # iclr preparing
    data_folder_iclr = "../data/"

    if 2018 <= int(year) <= 2021:
        papers_iclr = []
        invitations = set([])
        confidences_set = set([])
        ratings_set = set([])
        experience_assessments_set = set([])
        with jsonlines.open(data_folder_iclr + "iclr_%s.jsonl" % year) as reader:
            paper_list = []
            for line in reader:
                paper_list.append(line)
            print("ICLR paper count", len(paper_list))
            for paper in paper_list:
                paper_new = {}
                if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                              "") != "" and paper.get(
                    "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                             []) != []:
                    paper_new["paper_id"] = "iclr_" + year + "_" + paper["id"]
                    paper_new["paper_title"] = paper["title"]
                    paper_new["paper_abstract"] = paper["abstract"]
                    paper_new["paper_acceptance"] = paper["comment"]
                    final_decision_content = paper["final_decision"]["content"]
                    final_decision_time = paper["final_decision"]["tmdate"]
                    if year == "2019":
                        final_decision = final_decision_content["metareview"]  # it should be metareview when in 2019
                    else:
                        final_decision = final_decision_content["comment"]
                    if final_decision != "":
                        paper_new["meta_review"] = final_decision

                        # if y == "2020":
                        for rc in paper["reviews_commments"]:
                            # if rc["replyto"] == paper["id"]:
                            for s in rc["signatures"]:
                                invitations.add(rc["invitation"].split("/")[-1] + "#" + s.split("/")[-1])
                            # if rc["invitation"].split("/")[-1] == "Comment":
                            #     print(paper["title"])

                        reviews = []
                        ratings = []
                        confidences = []
                        official_reviews = []
                        for rc in paper["reviews_commments"]:
                            if time.localtime(final_decision_time / 1000) >= time.localtime(rc["tcdate"] / 1000):
                                review = {}
                                review["review_id"] = rc["id"]
                                if rc["replyto"] == paper["id"]:
                                    review["reply_to"] = paper_new["paper_id"]
                                else:
                                    review["reply_to"] = rc["replyto"]

                                cs = rc["content"]
                                if "review" in cs.keys():
                                    review["comment"] = cs["review"]
                                if "comment" in cs.keys():
                                    review["comment"] = cs["comment"]
                                if "ethics_review" in cs.keys():
                                    review["comment"] = review.get("comment", "") + cs["ethics_review"]
                                if "rating" in cs.keys():
                                    review["rating"] = cs["rating"]
                                    if isinstance(review["rating"], int):
                                        ratings.append(review["rating"])
                                    else:
                                        s = int(review["rating"].split(":")[0].strip())
                                        ratings.append(s)
                                        review["rating"] = s
                                    official_reviews.append(rc["id"])
                                if "confidence" in cs.keys():
                                    review["confidence"] = cs["confidence"]
                                if "experience_assessment" in cs.keys():
                                    experience_assessment = cs["experience_assessment"]
                                    experience_assessments_set.add(experience_assessment)
                                    if experience_assessment == "I have read many papers in this area.":
                                        review["confidence"] = 3
                                    if experience_assessment == "I have published one or two papers in this area.":
                                        review["confidence"] = 4
                                    if experience_assessment == "I have published in this field for several years.":
                                        review["confidence"] = 5
                                    if experience_assessment == "I do not know much about this area.":
                                        review["confidence"] = 1
                                if "confidence" in review.keys():
                                    if isinstance(review["confidence"], int):
                                        confidences.append(review["confidence"])
                                    else:
                                        s = int(review["confidence"].split(":")[0].strip())
                                        confidences.append(s)
                                        review["confidence"] = s

                                invit = rc["invitation"].split("/")[-1]
                                if "Official_Review" == invit or "Ethics_Review" == invit:
                                    review["writer"] = "official_reviewer"
                                elif "Official_Comment" == invit or "Comment" == invit:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    elif "AnonReviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Area_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Program_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    else:
                                        review["writer"] = "public"

                                elif "Public_Comment" == invit:
                                    review["writer"] = "public"
                                else:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    else:
                                        review["writer"] = "public"
                                if review.get("comment", "") != "":
                                    reviews.append(review)
                            else:
                                print(paper_new["paper_id"])
                                print(final_decision_time)
                                print(rc)
                        ratings_set.update(ratings)
                        confidences_set.update(confidences)

                        paper_new["reviews"] = reviews
                        if paper_new["meta_review"] != "" and len(official_reviews) > 0:
                            papers_iclr.append(paper_new)
                        else:
                            print(paper)

            # print(sorted(invitations))
        print("ICLR confidences", year, confidences_set)
        print("ICLR ratings", year, ratings_set)
        print("ICLR experience_assessment (only 2020)", year, experience_assessments_set)
        split_data(papers_iclr)
        return papers_iclr
    elif int(year) == 2022:
        papers_iclr = []
        invitations = set([])
        confidences_set = set([])
        ratings_set = set([])
        with jsonlines.open(data_folder_iclr + "iclr_%s.jsonl" % year) as reader:
            paper_list = []
            for line in reader:
                paper_list.append(line)
            print("ICLR paper count", len(paper_list))
            for paper in paper_list:
                # if paper["title"] == "Decoupled Contrastive Learning":
                #     print(paper)
                #     break
                paper_new = {}
                if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                              []) != [] and paper.get(
                    "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                             []) != []:
                    paper_new["paper_id"] = "iclr_" + year + "_" + paper["id"]
                    paper_new["paper_title"] = paper["title"]
                    paper_new["paper_abstract"] = paper["abstract"]
                    paper_new["paper_acceptance"] = paper["comment"]
                    final_decision_content = paper["final_decision"]["content"]
                    final_decision_time = paper["final_decision"]["tmdate"]
                    final_decision = final_decision_content["comment"]
                    if final_decision != "":
                        paper_new["meta_review"] = final_decision

                        # if y == "2020":
                        for rc in paper["reviews_commments"]:
                            # if rc["replyto"] == paper["id"]:
                            for s in rc["signatures"]:
                                invitations.add(rc["invitation"].split("/")[-1] + "#" + s.split("/")[-1])
                            # if rc["invitation"].split("/")[-1] == "Comment":
                            #     print(paper["title"])

                        reviews = []
                        ratings = []
                        confidences = []
                        official_reviews = []
                        for rc in paper["reviews_commments"]:
                            if time.localtime(final_decision_time / 1000) >= time.localtime(rc["tcdate"] / 1000):
                                review = {}
                                review["review_id"] = rc["id"]

                                invit = rc["invitation"].split("/")[-1]
                                if "Official_Review" == invit or "Ethics_Review" == invit:
                                    review["writer"] = "official_reviewer"
                                elif "Official_Comment" == invit or "Comment" == invit:
                                    signatures = " ".join(rc["signatures"])
                                    if "Authors" in signatures:
                                        review["writer"] = "author"
                                    elif "AnonReviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Reviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Area_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Program_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    else:
                                        review["writer"] = "public"

                                elif "Public_Comment" == invit:
                                    review["writer"] = "public"
                                else:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    else:
                                        review["writer"] = "public"

                                review["reply_to"] = rc["replyto"]
                                if rc["replyto"] == paper["id"]:
                                    review["reply_to"] = paper_new["paper_id"]

                                cs = rc["content"]
                                review_text = ""
                                if "review" in cs.keys():
                                    review_text = review_text + " " + cs["review"]
                                if "comment" in cs.keys():
                                    review_text = review_text + " " + cs["comment"]
                                if "main_review" in cs.keys():
                                    review_text = cs["main_review"]
                                if "summary_of_the_review" in cs.keys():
                                    review_text = review_text + " " + cs["summary_of_the_review"]
                                if "summary_of_the_paper" in cs.keys():
                                    review_text = cs["summary_of_the_paper"] + " " + review_text

                                if "ethics_review" in cs.keys():
                                    review_text = review_text + " " + cs["ethics_review"]
                                if "issues_acknowledged_description" in cs.keys():
                                    review_text = review_text + " " + cs["issues_acknowledged_description"]
                                if "ethics_review" in cs.keys() and "recommendation" in cs.keys():
                                    review_text = review_text + " " + cs["recommendation"]

                                review["comment"] = review_text
                                if "rating" in cs.keys():
                                    review["rating"] = cs["rating"]
                                if "recommendation" in cs.keys() and "ethics_review" not in cs.keys():
                                    review["rating"] = cs["recommendation"]
                                if "rating" in review.keys():
                                    official_reviews.append(rc["id"])
                                    if isinstance(review["rating"], int):
                                        ratings.append(review["rating"])
                                    else:
                                        s = int(review["rating"].split(":")[0].strip())
                                        ratings.append(s)
                                        review["rating"] = s
                                if "confidence" in cs.keys():
                                    review["confidence"] = cs["confidence"]
                                if "confidence" in review.keys():
                                    if isinstance(review["confidence"], int):
                                        confidences.append(review["confidence"])
                                    else:
                                        s = int(review["confidence"].split(":")[0].strip())
                                        confidences.append(s)
                                        review["confidence"] = s
                                if review.get("comment", "") != "":
                                    reviews.append(review)
                            else:
                                print(paper_new["paper_id"])
                                print(final_decision_time)
                                print(rc)

                        ratings_set.update(ratings)
                        confidences_set.update(confidences)

                        paper_new["reviews"] = reviews
                        if len(official_reviews) > 0:
                            papers_iclr.append(paper_new)
                # else:
                #     print(paper)

            # print(sorted(invitations))
        print("ICLR confidences", year, confidences_set)
        print("ICLR ratings", year, ratings_set)
        split_data(papers_iclr)
        return papers_iclr
    elif int(year) == 2023:
        papers_iclr = []
        invitations = set([])
        confidences_set = set([])
        ratings_set = set([])
        with jsonlines.open(data_folder_iclr + "iclr_%s.jsonl" % year) as reader:
            paper_list = []
            for line in reader:
                paper_list.append(line)
            print("ICLR paper count", len(paper_list))
            for paper in paper_list:
                # if paper["title"] == "Decoupled Contrastive Learning":
                #     print(paper)
                #     break
                paper_new = {}
                if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                              []) != [] and paper.get(
                    "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                             []) != []:
                    # print(paper)
                    paper_new["paper_id"] = "iclr_" + year + "_" + paper["id"]
                    paper_new["paper_title"] = paper["title"]
                    paper_new["paper_abstract"] = paper["abstract"]
                    final_decision_content = paper["final_decision"]["content"]
                    paper_new["paper_acceptance"] = final_decision_content["decision"]
                    final_decision_time = paper["final_decision"]["tmdate"]
                    final_decision = final_decision_content["metareview:_summary,_strengths_and_weaknesses"]
                    if final_decision != "":
                        paper_new["meta_review"] = final_decision

                        # if y == "2020":
                        for rc in paper["reviews_commments"]:
                            # if rc["replyto"] == paper["id"]:
                            for s in rc["signatures"]:
                                invitations.add(rc["invitation"].split("/")[-1] + "#" + s.split("/")[-1])
                            # if rc["invitation"].split("/")[-1] == "Comment":
                            #     print(paper["title"])

                        reviews = []
                        ratings = []
                        confidences = []
                        official_reviews = []
                        for rc in paper["reviews_commments"]:
                            if time.localtime(final_decision_time / 1000) >= time.localtime(rc["tcdate"] / 1000):
                                review = {}
                                review["review_id"] = rc["id"]

                                invit = rc["invitation"].split("/")[-1]
                                if "Official_Review" == invit or "Ethics_Review" == invit:
                                    review["writer"] = "official_reviewer"
                                elif "Official_Comment" == invit or "Comment" == invit:
                                    signatures = " ".join(rc["signatures"])
                                    if "Authors" in signatures:
                                        review["writer"] = "author"
                                    elif "AnonReviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Reviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Area_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Program_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    else:
                                        review["writer"] = "public"

                                elif "Public_Comment" == invit:
                                    review["writer"] = "public"
                                else:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    else:
                                        review["writer"] = "public"

                                review["reply_to"] = rc["replyto"]
                                if rc["replyto"] == paper["id"]:
                                    review["reply_to"] = paper_new["paper_id"]

                                cs = rc["content"]
                                review_text = ""
                                if "title" in cs.keys():
                                    review_text = review_text + " " + cs["title"]
                                if "review" in cs.keys():
                                    review_text = review_text + " " + cs["review"]
                                if "comment" in cs.keys():
                                    review_text = review_text + " " + cs["comment"]
                                if "main_review" in cs.keys():
                                    review_text = cs["main_review"]
                                if "strength_and_weaknesses" in cs.keys():
                                    review_text = review_text + " " + cs["strength_and_weaknesses"]
                                if "clarity,_quality,_novelty_and_reproducibility" in cs.keys():
                                    review_text = review_text + " " + cs["clarity,_quality,_novelty_and_reproducibility"]
                                if "summary_of_the_review" in cs.keys():
                                    review_text = review_text + " " + cs["summary_of_the_review"]
                                if "summary_of_the_paper" in cs.keys():
                                    review_text = cs["summary_of_the_paper"] + " " + review_text

                                if "ethics_review" in cs.keys():
                                    review_text = review_text + " " + cs["ethics_review"]
                                if "issues_acknowledged_description" in cs.keys():
                                    review_text = review_text + " " + cs["issues_acknowledged_description"]
                                if "ethics_review" in cs.keys() and "recommendation" in cs.keys():
                                    review_text = review_text + " " + cs["recommendation"]

                                review["comment"] = review_text
                                if "rating" in cs.keys():
                                    review["rating"] = cs["rating"]
                                if "recommendation" in cs.keys() and "ethics_review" not in cs.keys():
                                    review["rating"] = cs["recommendation"]
                                if "rating" in review.keys():
                                    official_reviews.append(rc["id"])
                                    if isinstance(review["rating"], int):
                                        ratings.append(review["rating"])
                                    else:
                                        s = int(review["rating"].split(":")[0].strip())
                                        ratings.append(s)
                                        review["rating"] = s
                                if "confidence" in cs.keys():
                                    review["confidence"] = cs["confidence"]
                                if "confidence" in review.keys():
                                    if isinstance(review["confidence"], int):
                                        confidences.append(review["confidence"])
                                    else:
                                        s = int(review["confidence"].split(":")[0].strip())
                                        confidences.append(s)
                                        review["confidence"] = s
                                if review.get("comment", "") != "":
                                    reviews.append(review)
                            else:
                                print(paper_new["paper_id"])
                                print(final_decision_time)
                                print(rc)

                        ratings_set.update(ratings)
                        confidences_set.update(confidences)

                        paper_new["reviews"] = reviews
                        if len(official_reviews) > 0:
                            papers_iclr.append(paper_new)
                # else:
                #     print(paper)

            # print(sorted(invitations))
        print("ICLR confidences", year, confidences_set)
        print("ICLR ratings", year, ratings_set)
        split_data(papers_iclr)
        return papers_iclr
    elif int(year) == 2024:
        papers_iclr = []
        invitations = set([])
        confidences_set = set([])
        ratings_set = set([])
        with jsonlines.open(data_folder_iclr + "iclr_%s.jsonl" % year) as reader:
            paper_list = []
            for line in reader:
                paper_list.append(line)
            print("ICLR paper count", len(paper_list))
            for paper in paper_list:
                # if paper["title"] == "Decoupled Contrastive Learning":
                #     print(paper)
                #     break
                paper_new = {}
                if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                              []) != [] and paper.get(
                    "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                             []) != []:
                    # print(paper)
                    paper_new["paper_id"] = "iclr_" + year + "_" + paper["id"]
                    paper_new["paper_title"] = paper["title"]
                    paper_new["paper_abstract"] = paper["abstract"]
                    final_decision_content = paper["final_decision"]["content"]
                    paper_new["paper_acceptance"] = final_decision_content["decision"]


                    # as there are a decision and a meta-review, we have to distinguish them
                    reviews_comments = []
                    for review in paper["reviews_commments"]:
                        if "metareview" in review["content"].keys():
                            final_decision = review["content"]["metareview"]["value"]
                            final_decision_time = paper["final_decision"]["cdate"]
                        else:
                            reviews_comments.append(review)


                    if final_decision != "":
                        paper_new["meta_review"] = final_decision

                        # if y == "2020":
                        for rc in paper["reviews_commments"]:
                            # if rc["replyto"] == paper["id"]:
                            for s in rc["signatures"]:
                                invitations.add(rc["invitations"][0].split("/")[-1] + "#" + s.split("/")[-1])
                            # if rc["invitation"].split("/")[-1] == "Comment":
                            #     print(paper["title"])

                        reviews = []
                        ratings = []
                        confidences = []
                        official_reviews = []
                        for rc in reviews_comments:
                            if time.localtime(final_decision_time / 1000) >= time.localtime(rc["cdate"] / 1000):
                                review = {}
                                review["review_id"] = rc["id"]

                                invit = rc["invitations"][0].split("/")[-1]
                                if "Official_Review" == invit or "Ethics_Review" == invit:
                                    review["writer"] = "official_reviewer"
                                elif "Official_Comment" == invit or "Comment" == invit:
                                    signatures = " ".join(rc["signatures"])
                                    if "Authors" in signatures:
                                        review["writer"] = "author"
                                    elif "AnonReviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Reviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Area_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Program_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    else:
                                        review["writer"] = "public"

                                elif "Public_Comment" == invit:
                                    review["writer"] = "public"
                                else:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    else:
                                        review["writer"] = "public"

                                review["reply_to"] = rc["replyto"]
                                if rc["replyto"] == paper["id"]:
                                    review["reply_to"] = paper_new["paper_id"]

                                cs = rc["content"]
                                review_text = ""
                                if "review" in cs.keys():
                                    review_text = review_text + " " + cs["review"]["value"]
                                if "comment" in cs.keys():
                                    review_text = review_text + " " + cs["comment"]["value"]
                                if "summary" in cs.keys():
                                    review_text = cs["summary"]["value"]
                                if "strengths" in cs.keys():
                                    review_text = review_text + " " + cs["strengths"]["value"]
                                if "weaknesses" in cs.keys():
                                    review_text = review_text + " " + cs["weaknesses"]["value"]

                                if "ethics_review" in cs.keys():
                                    review_text = review_text + " " + cs["ethics_review"]["value"]
                                if "issues_acknowledged_description" in cs.keys():
                                    review_text = review_text + " " + cs["issues_acknowledged_description"]["value"]
                                if "ethics_review" in cs.keys() and "recommendation" in cs.keys():
                                    review_text = review_text + " " + cs["recommendation"]["value"]

                                review["comment"] = review_text
                                if "rating" in cs.keys():
                                    review["rating"] = cs["rating"]["value"]
                                if "recommendation" in cs.keys() and "ethics_review" not in cs.keys():
                                    review["rating"] = cs["recommendation"]
                                if "rating" in review.keys():
                                    official_reviews.append(rc["id"])
                                    if isinstance(review["rating"], int):
                                        ratings.append(review["rating"])
                                    else:
                                        s = int(review["rating"].split(":")[0].strip())
                                        ratings.append(s)
                                        review["rating"] = s
                                if "confidence" in cs.keys():
                                    review["confidence"] = cs["confidence"]["value"]
                                if "confidence" in review.keys():
                                    if isinstance(review["confidence"], int):
                                        confidences.append(review["confidence"])
                                    else:
                                        s = int(review["confidence"].split(":")[0].strip())
                                        confidences.append(s)
                                        review["confidence"] = s
                                if review.get("comment", "") != "":
                                    reviews.append(review)
                            else:
                                print(paper_new["paper_id"])
                                print(final_decision_time)
                                print(rc)

                        ratings_set.update(ratings)
                        confidences_set.update(confidences)

                        paper_new["reviews"] = reviews
                        if len(official_reviews) > 0:
                            papers_iclr.append(paper_new)
                # else:
                #     print(paper)

            # print(sorted(invitations))
        print("ICLR confidences", year, confidences_set)
        print("ICLR ratings", year, ratings_set)
        split_data(papers_iclr)
        return papers_iclr
    else:
        print(f"There is no data for year {year}")
        return []


def prepare_nips(year):
    data_folder_nips = "../data/"

    if int(year) == 2021:
        papers_nips = []
        invitations = set([])
        confidences_set = set([])
        ratings_set = set([])
        with jsonlines.open(data_folder_nips + "nips_%s.jsonl" % year) as reader:
            paper_list = []
            for line in reader:
                paper_list.append(line)
        print("NIPS paper count", len(paper_list))
        for paper in paper_list:
            paper_new = {}

            if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get(
                    "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                             []) != []:
                paper_new["paper_id"] = "nips_" + year + "_" + paper["id"]
                paper_new["paper_title"] = paper["title"]
                paper_new["paper_abstract"] = paper["abstract"]
                paper_new["paper_acceptance"] = paper["comment"]
                final_decision_content = paper["final_decision"]["content"]
                final_decision_time = paper["final_decision"]["tmdate"]
                final_decision = final_decision_content["comment"]
                if final_decision != "":
                    paper_new["meta_review"] = final_decision

                    # if y == "2020":
                    for rc in paper["reviews_commments"]:
                        # if rc["replyto"] == paper["id"]:
                        for s in rc["signatures"]:
                            invitations.add(rc["invitation"].split("/")[-1] + "#" + s.split("/")[-1])
                        if rc["invitation"].split("/")[-1] == "Comment":
                            print(paper["title"])

                    reviews = []
                    confidences = []
                    ratings = []
                    official_reviews = []
                    for rc in paper["reviews_commments"]:
                        if time.localtime(final_decision_time / 1000) >= time.localtime(
                                rc["tcdate"] / 1000):  # tmdata is the modified time
                            review = {}
                            review["review_id"] = rc["id"]

                            invit = rc["invitation"].split("/")[-1]
                            if "Official_Review" == invit or "Ethics_Review" == invit:
                                review["writer"] = "official_reviewer"
                            elif "Official_Comment" == invit or "Comment" == invit:
                                signatures = " ".join(rc["signatures"])
                                if "Authors" in signatures:
                                    review["writer"] = "author"
                                elif "AnonReviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Reviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Area_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Program_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                else:
                                    review["writer"] = "public"

                            elif "Public_Comment" == invit:
                                review["writer"] = "public"
                            else:
                                signatures = " ".join(rc["signatures"])
                                if "Author" in signatures:
                                    review["writer"] = "author"
                                else:
                                    review["writer"] = "public"

                            review["reply_to"] = rc["replyto"]
                            if rc["replyto"] == paper["id"]:
                                review["reply_to"] = paper_new["paper_id"]

                            cs = rc["content"]
                            # print(cs)
                            review_text = ""
                            if "review" in cs.keys():
                                review_text = review_text + " " + cs["review"]
                            if "comment" in cs.keys():
                                review_text = review_text + " " + cs["comment"]
                            if "main_review" in cs.keys():
                                review_text = review_text + " " + cs["main_review"]
                            if "summary" in cs.keys():
                                review_text = cs["summary"] + " " + review_text
                            if "limitations_and_societal_impact" in cs.keys():
                                review_text = review_text + " " + cs["limitations_and_societal_impact"]

                            if "ethics_review" in cs.keys():
                                review_text = review_text + " " + cs["ethics_review"]
                            if "issues_acknowledged_description" in cs.keys():
                                review_text = review_text + " " + cs["issues_acknowledged_description"]
                            if "ethics_review" in cs.keys() and "recommendation" in cs.keys():
                                review_text = review_text + " " + cs["recommendation"]

                            review["comment"] = review_text
                            if "rating" in cs.keys():
                                review["rating"] = cs["rating"]
                                official_reviews.append(rc["id"])
                            if "rating" in review.keys():
                                if isinstance(review["rating"], int):
                                    ratings.append(review["rating"])
                                else:
                                    s = int(review["rating"].split(":")[0].strip())
                                    ratings.append(s)
                                    review["rating"] = s
                            if "recommendation" in cs.keys() and "ethics_review" not in cs.keys():
                                review["rating"] = cs["recommendation"]
                            if "confidence" in cs.keys():
                                review["confidence"] = cs["confidence"]
                            if "confidence" in review.keys():
                                if isinstance(review["confidence"], int):
                                    confidences.append(review["confidence"])
                                else:
                                    s = int(review["confidence"].split(":")[0].strip())
                                    confidences.append(s)
                                    review["confidence"] = s
                            if review.get("comment", "") != "":
                                reviews.append(review)
                        else:
                            print(paper_new["paper_id"])
                            print(final_decision_time)
                            print(rc)

                    ratings_set.update(ratings)
                    confidences_set.update(confidences)

                    paper_new["reviews"] = reviews
                    if len(official_reviews) > 0:
                        papers_nips.append(paper_new)
                    else:
                        print(paper)
            # else:
            #     print(paper["title"])
        # print(sorted(invitations))
        print("NIPS confidences", year, confidences_set)
        print("NIPS ratings", year, ratings_set)
        split_data(papers_nips)
        return papers_nips
    elif int(year) == 2022:
        papers_nips = []
        invitations = set([])
        confidences_set = set([])
        ratings_set = set([])
        with jsonlines.open(data_folder_nips + "nips_%s.jsonl" % year) as reader:
            paper_list = []
            for line in reader:
                paper_list.append(line)
        print("NIPS paper count", len(paper_list))
        for paper in paper_list:
            paper_new = {}

            if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get(
                    "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                             []) != []:
                paper_new["paper_id"] = "nips_" + year + "_" + paper["id"]
                paper_new["paper_title"] = paper["title"]
                paper_new["paper_abstract"] = paper["abstract"]
                paper_new["paper_acceptance"] = paper["comment"]
                final_decision_content = paper["final_decision"]["content"]
                # print(final_decision_content.keys())
                final_decision_time = paper["final_decision"]["tmdate"]
                final_decision = final_decision_content["metareview"]
                if final_decision != "":
                    paper_new["meta_review"] = final_decision
                    # print(paper_new["paper_id"], len(paper["reviews_commments"]))
                    reviews = []
                    confidences = []
                    ratings = []
                    official_reviews = []
                    for rc in paper["reviews_commments"]:
                        if time.localtime(final_decision_time / 1000) >= time.localtime(rc["tcdate"] / 1000):
                            review = {}
                            review["review_id"] = rc["id"]

                            invit = rc["invitation"].split("/")[-1]
                            # print(invit)
                            if "Official_Review" == invit or "Ethics_Review" == invit:
                                review["writer"] = "official_reviewer"
                            elif "Official_Comment" == invit or "Comment" == invit:
                                signatures = " ".join(rc["signatures"])
                                if "Authors" in signatures:
                                    review["writer"] = "author"
                                elif "AnonReviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Reviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Area_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Program_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                else:
                                    review["writer"] = "public"

                            elif "Public_Comment" == invit:
                                review["writer"] = "public"
                            else:
                                signatures = " ".join(rc["signatures"])
                                if "Author" in signatures:
                                    review["writer"] = "author"
                                else:
                                    review["writer"] = "public"

                            review["reply_to"] = rc["replyto"]
                            if rc["replyto"] == paper["id"]:
                                review["reply_to"] = paper_new["paper_id"]

                            cs = rc["content"]
                            # print(cs.keys())
                            review_text = ""
                            if "summary" in cs.keys():
                                review_text = review_text + " " + cs["summary"]
                            if "strengths_and_weaknesses" in cs.keys():
                                review_text = review_text + " " + cs["strengths_and_weaknesses"]
                            if "questions" in cs.keys():
                                review_text = review_text + " " + cs["questions"]
                            if "limitations" in cs.keys():
                                review_text = review_text + " " + cs["limitations"]
                            if "comment" in cs.keys():
                                review_text = review_text + " " + cs["comment"]

                            if "ethics_review" in cs.keys():
                                review_text = review_text + " " + cs["ethics_review"]
                            if "issues_acknowledged_description" in cs.keys():
                                review_text = review_text + " " + cs["issues_acknowledged_description"]
                            if "ethics_review" in cs.keys() and "recommendation" in cs.keys():
                                review_text = review_text + " " + cs["recommendation"]

                            review["comment"] = review_text
                            if "rating" in cs.keys():
                                review["rating"] = cs["rating"]
                                official_reviews.append(rc["id"])
                            if "rating" in review.keys():
                                if isinstance(review["rating"], int):
                                    ratings.append(review["rating"])
                                else:
                                    s = int(review["rating"].split(":")[0].strip())
                                    ratings.append(s)
                                    review["rating"] = s
                            if "confidence" in cs.keys():
                                review["confidence"] = cs["confidence"]
                            if "confidence" in review.keys():
                                if isinstance(review["confidence"], int):
                                    confidences.append(review["confidence"])
                                else:
                                    s = int(review["confidence"].split(":")[0].strip())
                                    confidences.append(s)
                                    review["confidence"] = s
                            if review.get("comment", "") != "":
                                reviews.append(review)
                        else:
                            print(paper_new["paper_id"])
                            print(final_decision_time)
                            print(rc)

                    ratings_set.update(ratings)
                    confidences_set.update(confidences)

                    paper_new["reviews"] = reviews
                    if len(official_reviews) > 0:
                        papers_nips.append(paper_new)
                    else:
                        print(paper)
            # else:
            #     print(paper["title"])
        # print(sorted(invitations))
        print("NIPS confidences", year, confidences_set)
        print("NIPS ratings", year, ratings_set)
        split_data(papers_nips)
        return papers_nips
    elif int(year) == 2023:
        papers_nips = []
        invitations = set([])
        confidences_set = set([])
        ratings_set = set([])
        with jsonlines.open(data_folder_nips + "nips_%s.jsonl" % year) as reader:
            paper_list = []
            for line in reader:
                paper_list.append(line)
        print("NIPS paper count", len(paper_list))
        for paper in paper_list:
            # if paper["title"] == "Decoupled Contrastive Learning":
            #     print(paper)
            #     break
            paper_new = {}
            if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                          []) != [] and paper.get(
                "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                         []) != []:
                # print(paper)
                paper_new["paper_id"] = "nips_" + year + "_" + paper["id"]
                paper_new["paper_title"] = paper["title"]
                paper_new["paper_abstract"] = paper["abstract"]
                final_decision_content = paper["final_decision"]["content"]
                paper_new["paper_acceptance"] = final_decision_content["decision"]

                # as there are a decision and a meta-review, we have to distinguish them
                final_decision = final_decision_content["comment"]["value"]
                final_decision_time = paper["final_decision"]["cdate"]
                if final_decision != "":
                    paper_new["meta_review"] = final_decision

                    # if y == "2020":
                    for rc in paper["reviews_commments"]:
                        # if rc["replyto"] == paper["id"]:
                        for s in rc["signatures"]:
                            invitations.add(rc["invitations"][0].split("/")[-1] + "#" + s.split("/")[-1])
                        # if rc["invitation"].split("/")[-1] == "Comment":
                        #     print(paper["title"])

                    reviews = []
                    ratings = []
                    confidences = []
                    official_reviews = []
                    for rc in paper["reviews_commments"]:
                        if time.localtime(final_decision_time / 1000) >= time.localtime(rc["cdate"] / 1000):
                            review = {}
                            review["review_id"] = rc["id"]

                            invit = rc["invitations"][0].split("/")[-1]
                            if "Official_Review" == invit or "Ethics_Review" == invit:
                                review["writer"] = "official_reviewer"
                            elif "Official_Comment" == invit or "Comment" == invit:
                                signatures = " ".join(rc["signatures"])
                                if "Authors" in signatures:
                                    review["writer"] = "author"
                                elif "AnonReviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Reviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Area_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Program_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                else:
                                    review["writer"] = "public"

                            elif "Public_Comment" == invit:
                                review["writer"] = "public"
                            else:
                                signatures = " ".join(rc["signatures"])
                                if "Author" in signatures:
                                    review["writer"] = "author"
                                else:
                                    review["writer"] = "public"

                            review["reply_to"] = rc["replyto"]
                            if rc["replyto"] == paper["id"]:
                                review["reply_to"] = paper_new["paper_id"]

                            cs = rc["content"]
                            review_text = ""
                            if "review" in cs.keys():
                                review_text = review_text + " " + cs["review"]["value"]
                            if "comment" in cs.keys():
                                review_text = review_text + " " + cs["comment"]["value"]
                            if "summary" in cs.keys():
                                review_text = cs["summary"]["value"]
                            if "strengths" in cs.keys():
                                review_text = review_text + " " + cs["strengths"]["value"]
                            if "weaknesses" in cs.keys():
                                review_text = review_text + " " + cs["weaknesses"]["value"]
                            if "limitations" in cs.keys():
                                review_text = review_text + " " + cs["limitations"]["value"]
                            if "rebuttal" in cs.keys():
                                review_text = review_text + " " + cs["rebuttal"]["value"]

                            if "ethics_review" in cs.keys():
                                review_text = review_text + " " + cs["ethics_review"]["value"]
                            if "issues_acknowledged_description" in cs.keys():
                                review_text = review_text + " " + cs["issues_acknowledged_description"]["value"]
                            if "ethics_review" in cs.keys() and "recommendation" in cs.keys():
                                review_text = review_text + " " + cs["recommendation"]["value"]

                            review["comment"] = review_text
                            if "rating" in cs.keys():
                                review["rating"] = cs["rating"]["value"]
                            if "recommendation" in cs.keys() and "ethics_review" not in cs.keys():
                                review["rating"] = cs["recommendation"]
                            if "rating" in review.keys():
                                official_reviews.append(rc["id"])
                                if isinstance(review["rating"], int):
                                    ratings.append(review["rating"])
                                else:
                                    s = int(review["rating"].split(":")[0].strip())
                                    ratings.append(s)
                                    review["rating"] = s
                            if "confidence" in cs.keys():
                                review["confidence"] = cs["confidence"]["value"]
                            if "confidence" in review.keys():
                                if isinstance(review["confidence"], int):
                                    confidences.append(review["confidence"])
                                else:
                                    s = int(review["confidence"].split(":")[0].strip())
                                    confidences.append(s)
                                    review["confidence"] = s
                            if review.get("comment", "") != "":
                                reviews.append(review)
                            else:
                                print(paper_new["paper_id"])
                                print(rc)
                        else:
                            print(paper_new["paper_id"])
                            print(final_decision_time)
                            print(rc)

                    ratings_set.update(ratings)
                    confidences_set.update(confidences)

                    paper_new["reviews"] = reviews
                    if len(official_reviews) > 0:
                        papers_nips.append(paper_new)
            # else:
            #     print(paper)
        # print(sorted(invitations))
        print("NIPS confidences", year, confidences_set)
        print("NIPS ratings", year, ratings_set)
        split_data(papers_nips)
        return papers_nips
    else:
        print(f"There is no data for year {year}")
        return []


if __name__ == "__main__":
    papers = []
    # 2018, 2025
    for iclr in range(2018, 2025):
        print("Year", iclr)
        papers_iclr = prepare_iclr(str(iclr))
        print("ICLR %d" % iclr, len(papers_iclr))
        papers.extend(papers_iclr)
    # 2021, 2024
    for nips in range(2021, 2024):
        print("Year", nips)
        papers_nips = prepare_nips(str(nips))
        print("NIPS %d" % nips, len(papers_nips))
        papers.extend(papers_nips)

    # validating the dataset
    papers_new = []
    for paper in papers:
        if paper.get("paper_id", "") == "":
            print("####")
        if paper.get("paper_title", "") == "":
            print("####")
        if paper.get("paper_abstract", "") == "":
            print("####")
        if paper.get("paper_acceptance", "") == "":
            print("####")
        if paper.get("meta_review", "") == "":
            print("####")
        if len(paper.get("reviews", [])) == 0:
            print("####")

        reviews_new = []
        for doc in paper["reviews"]:
            if doc.get("review_id", "") == "":
                print("review id")
            if doc.get("writer", "") == "":
                print("writer")
            if doc.get("comment", "") == "":
                print("comment")
            if doc.get("reply_to", "") == "":
                print("reply_to")
            doc["rating"] = doc.get("rating", -1)
            doc["confidence"] = doc.get("confidence", -1)
            reviews_new.append(doc)

        # remove reviews whose parent node cannot be found
        review_ids = []
        review_ids.append(paper["paper_id"])
        for review in reviews_new:
            review_ids.append(review["review_id"])

        reviews_error = []
        for review in reviews_new:
            if review["reply_to"] not in review_ids:
                reviews_error.append(review["review_id"])
        reviews_error_current = reviews_error
        while len(reviews_error_current) > 0:
            tmp = []
            for review in reviews_new:
                if review["reply_to"] in reviews_error_current:
                    tmp.append(review["review_id"])
                    reviews_error.append(review["review_id"])
            reviews_error_current = tmp

        reviews_tree = []
        for review in reviews_new:
            if review["review_id"] not in reviews_error:
                reviews_tree.append(review)

        paper["reviews"] = reviews_tree

        papers_new.append(paper)

    # saving the dataset in jsonlines
    print("all papers", len(papers_new))
    with open("../data/peermeta_all.json", "w") as f:
        json.dump(papers_new, f, indent=4)
