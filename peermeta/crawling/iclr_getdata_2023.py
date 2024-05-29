# The first step is getting the paper list from the OpenReview getting data, and then get reviews with forum ids
import jsonlines
import openreview
import time
from tqdm import tqdm

# API V1
base_url = "https://api.openreview.net"
client = openreview.Client(baseurl=base_url)
notes = client.get_all_notes(signature='ICLR.cc/2023/Conference')  # using signature to get all submissions
print(len(notes))
print("all papers", len(notes))

papers = []
count = 0
for note in tqdm(notes):
    print(note)
    paper = {}
    paper["link"] = "https://openreview.net/forum?id=" + note.forum
    content = note.content
    paper["title"] = content['title']
    paper["authors"] = content['authors']
    paper["abstract"] = content['abstract']
    paper["tl_dr"] = content.get('TL;DR', "")
    paper["keywords"] = content['keywords']
    paper["id"] = note.forum
    paper["venue"] = content['venueid']
    if "ICLR.cc/2023/Conference/Desk_Rejected_Submission" != paper["venue"]:
        paper["pdf"] = "https://openreview.net" + content.get("pdf", "")
        rcs = client.get_notes(forum=paper["id"])
        # print(rcs)
        reviews_commments = []
        for rc in rcs:
            count += 1
            if count % 60 == 0:
                time.sleep(60)

            print(rc.to_json())
            decision_note = False
            if "title" in rc.content.keys():
                if rc.content["title"] == "Paper Decision":
                    decision_note = True
                    paper["final_decision"] = rc.to_json()
                    paper["number"] = len(rcs) - 2

            if not decision_note and paper["id"] != rc.id:
                reviews_commments.append(rc.to_json())
        print("reviews_comments", len(reviews_commments), len(rcs))
        paper["reviews_commments"] = reviews_commments

        papers.append(paper)

print("Final", len(papers))
with jsonlines.open("../data/iclr_2023.jsonl", "w") as writer:
    writer.write_all(papers)
