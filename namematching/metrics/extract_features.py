def jaccard(name1, name2):
    """
    Computes Jaccard similarity between two strings.

    Args:
        name1 (str): First string.
        name2 (str): Second string.

    Returns:
        float: Jaccard similarity score (between 0 and 1).
    """
    # Tokenize both names by splitting on whitespace
    tokens_1 = set(name1.split())
    tokens_2 = set(name2.split())

    # Compute the intersection of the two token sets (common words)
    intersection = tokens_1 & tokens_2

    # Compute the union of both token sets (all unique words from both)
    union = tokens_1 | tokens_2

    # Number of shared tokens divided by total unique tokens
    return len(intersection) / len(union) if union else 0
