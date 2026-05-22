import numpy as np

from moderation.text_checker import _scores_from_logits


def test_single_logit_low_score_is_not_always_one():
    """Regression: softmax on a single logit always yielded toxic=1.0."""
    scores = _scores_from_logits(np.array([-4.0]), {0: "toxic"})
    assert scores["toxic"] < 0.05


def test_single_logit_high_score_when_toxic():
    scores = _scores_from_logits(np.array([4.0]), {0: "toxic"})
    assert scores["toxic"] > 0.9


def test_multilabel_sigmoid_per_head():
    logits = np.array([-4.0, 4.0, -2.0])
    labels = {0: "toxic", 1: "severe_toxic", 2: "insult"}
    scores = _scores_from_logits(logits, labels)
    assert scores["toxic"] < 0.1
    assert scores["severe_toxic"] > 0.9
    assert scores["insult"] < 0.2


def test_multilabel_seven_heads():
    """oleksiizirka-style: toxic … identity_hate + none."""
    logits = np.array([-3.0, -3.0, 4.0, -3.0, -3.0, -3.0, -3.0])
    labels = {
        0: "toxic",
        1: "severe_toxic",
        2: "obscene",
        3: "threat",
        4: "insult",
        5: "identity_hate",
        6: "none",
    }
    scores = _scores_from_logits(logits, labels)
    assert scores["obscene"] > 0.9
    assert scores["none"] < 0.1
