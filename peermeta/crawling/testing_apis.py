def get_forum_v1():
    import openreview

    base_url = "https://api.openreview.net"
    client = openreview.Client(baseurl=base_url)
    rcs = client.get_notes(
            forum="7YfHla7IxBJ")
    for rc in rcs:
        print(rc)
    print()

    year = 2023
    base_url = "https://api.openreview.net"
    client = openreview.Client(baseurl=base_url)
    notes = client.get_all_notes(signature='ICLR.cc/%s/Conference' % year)  # using signature to get all submissions
    print(len(notes))


def get_all_papers_v2():
    import openreview

    # API V2
    client = openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username="miaoli.cs@gmail.com",
        password="limiao5002361995"
    )

    venue_group = client.get_group('ICLR.cc/2023/Conference')
    submission_name = venue_group.content['submission_name']['value']
    print(submission_name)
    notes = client.get_all_notes(content={"venueid": "ICLR.cc/2023/Conference"})
    print(len(notes))

if __name__ == "__main__":
    # get_all_papers_v2()
    get_forum_v1()