import pytest
from namematching import compare_names

# Positive name pairs — expected to be recognized as matches (high score)
POSITIVE_PAIRS = [
    ("louis larsonneur", "lou k larsonneur"),
    ("louis larsonneur", "lou k larsonneur jr"),
    ("louis larsonneur", "michel kaelig larsonneur jr"),
    ("a mitch mcconnell", "abraham mcconnell"),
    ("a mitch mcconnell", "mitchel mcconnell"),
    ("j k rowling", "joanne kathleen rowling"),
]

# Negative name pairs — expected to be recognized as non-matches (low score)
NEGATIVE_PAIRS = [
    ("louis larsonneur", "michel larsonneur"),
    ("michel smith", "michel larsonneur"),
    ("a smith", "abraham jones"),
    ("j doe", "jane smith"),
    ("john smith", "jane doe"),
    ("pierre dupont", "marie curie"),
    ("michel durand", "sophie martin"),
    ("emmanuel macron", "nicolas sarkozy"),
    ("serena williams", "roger federer"),
    ("bill gates", "elon musk"),
]


@pytest.mark.parametrize("name1, name2", POSITIVE_PAIRS)
def test_positive_pairs(name1, name2):
    """
    Positive test cases:
    Similar or variant forms of the same person should yield
    a high similarity score (> 0.5).
    """
    score = compare_names(name1, name2)
    print(f"[POS] {name1} <> {name2} -> {score:.4f}")  # Always printed with -s
    assert score > 0.5, f"Expected high similarity but got {score:.4f}"


@pytest.mark.parametrize("name1, name2", NEGATIVE_PAIRS)
def test_negative_pairs(name1, name2):
    """
    Negative test cases:
    Clearly different individuals should yield
    a low similarity score (< 0.5).
    """
    score = compare_names(name1, name2)
    print(f"[NEG] {name1} <> {name2} -> {score:.4f}")  # Always printed with -s
    assert score < 0.5, f"Expected low similarity but got {score:.4f}"
