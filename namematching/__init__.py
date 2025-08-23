# namematching/__init__.py

"""NameMatching - AI-based name comparison system."""

from namematching.matcher import NameMatcher
from utils import normalize_name

__version__ = "1.0"
__all__ = ["NameMatcher", "normalize_name", "compare_names"]
_MATCHER = None


def get_matcher():
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = NameMatcher()
    return _MATCHER


def compare_names(name1, name2):
    """
    Convenience function to quickly compare two names using the default model.

    Args:
        name1 (str): First name string.
        name2 (str): Second name string.
        model_path (str | Path, optional): Optional path to a trained model.
            If not provided, the default exported model is used.

    Returns:
        float: Similarity score between 0 and 1.
    """
    matcher = get_matcher()
    return matcher.similarity(name1, name2)
