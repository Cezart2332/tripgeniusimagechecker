from moderation.profanity import contains_profanity


def test_detects_fuck():
    assert contains_profanity("fuck")
    assert contains_profanity("What the fuck is this")


def test_allows_salut():
    assert not contains_profanity("Salut")
    assert not contains_profanity("hello team")


def test_detects_romanian_muie():
    assert contains_profanity("du-te in muie")
