from moderation.profanity import contains_profanity, normalize_for_profanity


def test_romanian_phrase_blocked():
    assert contains_profanity("sugi pula")
    assert contains_profanity("Sugeti pula")


def test_clean_text_allowed():
    assert not contains_profanity("Beautiful trail in the mountains")


def test_normalize_strips_diacritics():
    assert normalize_for_profanity("Sugeți pula") == "sugeti pula"
