def get_forum_v1():
    import openreview

    base_url = "https://api.openreview.net"
    client = openreview.Client(baseurl=base_url)
    rcs = client.get_notes(
            forum="x9jS8pX3dkx")
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

    venue_group = client.get_group('NeurIPS.cc/2023/Conference')
    submission_name = venue_group.content['submission_name']['value']
    print(submission_name)
    submissions = client.get_all_notes(invitation=f'NeurIPS.cc/2023/Conference/-/{submission_name}')
    print(len(submissions))

if __name__ == "__main__":
    get_all_papers_v2()