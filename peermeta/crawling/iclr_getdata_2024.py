# The first step is getting the paper list from the OpenReview getting data, and then get reviews with forum ids
import jsonlines
import openreview
import time
from tqdm import tqdm

# API V2
client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username="miaoli.cs@gmail.com",
    password="limiao5002361995"
    )

venue_group = client.get_group('ICLR.cc/2024/Conference')
submission_name = venue_group.content['submission_name']['value']
print(submission_name)
notes = client.get_all_notes(invitation=f'ICLR.cc/2024/Conference/-/{submission_name}')
print("all papers", len(notes))

papers = []
count = 0
for note in tqdm(notes):
    print(note)
    paper = {}
    paper["link"] = "https://openreview.net/forum?id=" + note.forum
    content = note.content
    paper["title"] = content['title']["value"]
    paper["authors"] = content['authors']["value"]
    paper["abstract"] = content['abstract']["value"]
    paper["tl_dr"] = content.get('TLDR', {"value": ""})["value"]
    paper["keywords"] = content['keywords']["value"]
    paper["id"] = note.forum
    paper["pdf"] = "https://openreview.net" + content["pdf"]["value"]
    paper["venue"] = content['venueid']["value"]

    if "ICLR.cc/2024/Conference/Desk_Rejected_Submission" != paper["venue"]:
        rcs = client.get_notes(forum=paper["id"])
        reviews_commments = []
        for rc in rcs:
            print(rc.to_json())
            decision_note = False
            if "title" in rc.content.keys():
                if rc.content["title"]["value"] == "Paper Decision":
                    decision_note = True
                    paper["final_decision"] = rc.to_json()
                    paper["number"] = len(rcs) - 2

            if not decision_note and paper["id"] != rc.id:
                reviews_commments.append(rc.to_json())
        print("reviews_comments", len(reviews_commments), len(rcs))
        paper["reviews_commments"] = reviews_commments

        papers.append(paper)
        count += 1
        if count % 60 == 0:
            time.sleep(30)

print("Final", len(papers))
with jsonlines.open("../data/iclr_2024.jsonl", "w") as writer:
    writer.write_all(papers)
