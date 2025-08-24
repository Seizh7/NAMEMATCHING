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

# Symmetry test pairs — comprehensive test for order independence
SYMMETRY_TEST_PAIRS = [
    # Original problematic case
    ("michael f bennet", "a mitchell mcconnell"),
    # Similar names with abbreviations
    ("a mitch mcconnell", "abraham mcconnell"),
    ("j k rowling", "joanne kathleen rowling"),
    ("louis larsonneur", "lou k larsonneur"),
    ("john smith", "j smith"),
    ("bill gates", "william gates"),
    ("f scott fitzgerald", "francis fitzgerald"),
    ("a lincoln", "abraham lincoln"),
    ("r j smith", "robert john smith"),
    # Different names (should have low but symmetric scores)
    ("pierre dupont", "marie curie"),
    ("jane doe", "j doe"),
    ("marie curie", "m curie"),
    ("albert einstein", "a einstein"),
    ("leonardo da vinci", "l da vinci"),
    # More diverse cases
    ("emmanuel macron", "e macron"),
    ("barack obama", "b obama"),
    ("donald trump", "d trump"),
    ("joe biden", "joseph biden"),
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


@pytest.mark.parametrize("name1, name2", SYMMETRY_TEST_PAIRS)
def test_symmetry(name1, name2):
    """
    Symmetry test cases:
    The similarity score should be identical regardless of argument order.
    compare_names(A, B) should equal compare_names(B, A).
    """
    score_ab = compare_names(name1, name2)
    score_ba = compare_names(name2, name1)
    
    print(f"[SYM] {name1} <> {name2} -> {score_ab:.6f}")
    print(f"[SYM] {name2} <> {name1} -> {score_ba:.6f}")
    print(f"[SYM] Difference: {abs(score_ab - score_ba):.6f}")
    
    # Allow for very small floating point differences (< 0.001)
    tolerance = 0.001
    assert abs(score_ab - score_ba) < tolerance, (
        f"Symmetry violation: {name1} <> {name2} = {score_ab:.6f}, "
        f"but {name2} <> {name1} = {score_ba:.6f}. "
        f"Difference: {abs(score_ab - score_ba):.6f} >= {tolerance}"
    )


def test_specific_symmetry_case():
    """
    Test the original problematic case specifically.
    """
    name1 = "michael f bennet"
    name2 = "a mitchell mcconnell"
    
    score1 = compare_names(name1, name2)
    score2 = compare_names(name2, name1)
    
    print("[SPECIFIC] Original problem case:")
    print(f"[SPECIFIC] {name1} -> {name2}: {score1:.8f}")
    print(f"[SPECIFIC] {name2} -> {name1}: {score2:.8f}")
    print(f"[SPECIFIC] Difference: {abs(score1 - score2):.8f}")
    
    # The model should be symmetric now
    assert abs(score1 - score2) < 0.001, (
        f"Model still asymmetric! {score1:.8f} vs {score2:.8f}"
    )
