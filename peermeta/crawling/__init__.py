# Step 1, Raw data crawled from websites, stored in the data folder
# The official OpenReview API: https://openreview-py.readthedocs.io/en/latest/
# ICLR: https://openreview.net/group?id=ICLR.cc (more reviews for other conferences could be crawled), ICLR 2018-2022
# NeurIPS: https://proceedings.neurips.cc/ NIPS 2021-2022
# there are no review ratings, or confidences for ICLR 2017, NIPS 2019, NIPS 2020

# Step 2, Combine all the data into 'peersum_all' from different conferences in different years, not only source documents but also summaries for summarization

# Structure of PeerSum
# paper_id: str
# paper_title: str
# paper_abstract, str
# paper_acceptance, str
# meta_review, str
# reviews, [{review_id, writer, comment, rating, confidence, reply_to}] (all reviews and comments)
# label, str, (train, val, test)

# please note:
# confidence: 1-5, int
# rating: 1-10, int
# writer: str, (author, official_reviewer, public)