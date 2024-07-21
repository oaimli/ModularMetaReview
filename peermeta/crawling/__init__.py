# Step 1, Raw data crawled from websites, stored in the data folder
# The official OpenReview API v2: https://openreview-py.readthedocs.io/en/latest/
# ICLR: https://openreview.net/group?id=ICLR.cc (more reviews for other conferences could be crawled), ICLR 2018-2024
# NeurIPS: https://proceedings.neurips.cc/ NIPS 2021-2023

# Step 2, Combine all the data into 'peermeta_all' from different conferences in different years

# Structure of PeerMeta
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