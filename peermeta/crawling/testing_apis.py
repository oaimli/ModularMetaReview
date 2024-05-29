def get_forum_v1():
    import openreview

    base_url = "https://api.openreview.net"
    client = openreview.Client(baseurl=base_url)
    rcs = client.get_notes(
            forum="7YfHla7IxBJ")
    for rc in rcs:
        print(rc)
    print()


def get_all_papers_v2():
    import openreview

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
    print(len(notes))

if __name__ == "__main__":
    get_all_papers_v2()
    # get_forum_v1()