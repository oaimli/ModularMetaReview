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

# Raw samples
# iclr_2018: 994
# iclr_2019: 1579
# iclr_2020: 2594
# iclr_2021: 3014
# iclr_2022: 3422
# iclr_2023: 4937
# iclr_2024: 7351
# nips_2021: 2334
# nips_2022: 2824
# nips_2023: 3395

# Valid samples: all papers 28028 (22420/2799/2809)