# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""NameMatching - AI-based name comparison system."""

from namematching.matcher import NameMatcher
from namematching.utils import normalize_name

__version__ = "1.0"
__all__ = ["NameMatcher", "normalize_name", "compare_names"]
_MATCHER = None


def get_matcher():
    """Get or create a singleton NameMatcher instance."""
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = NameMatcher()
    return _MATCHER


def compare_names(name1, name2):
    """
    Convenience function to quickly compare two names using the default model.

    This is the main entry point for the namematching package. It uses a
    pre-trained AI model to compute similarity between two person names.

    Args:
        name1 (str): First name string.
        name2 (str): Second name string.

    Returns:
        float: Similarity score between 0 and 1, where 1 means identical
               and 0 means completely different.

    Example:
        >>> from namematching import compare_names
        >>> similarity = compare_names("John Smith", "J. Smith")
        >>> print(f"Similarity: {similarity:.2f}")
        Similarity: 0.95
    """
    matcher = get_matcher()
    return matcher.similarity(name1, name2)
