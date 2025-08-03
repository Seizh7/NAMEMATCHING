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


def longest_common_substring(name1, name2):
    """
    Computes the length of the longest common substring.

    Args:
        name1 (str): First string.
        name2 (str): Second string.

    Returns:
        float: Normalized longest common substring length (0 to 1).
    """
    len_name1, len_name2 = len(name1), len(name2)
    max_len = 0

    # Initialize a 2D matrix to store LCS
    # Dimensions: (len_name1 + 1) rows × (len_name2 + 1) columns
    lcs_table = [
        [0] * (len_name2 + 1)
        for _ in range(len_name1 + 1)
    ]

    # Loop over each character of name1 and name2
    for i in range(len_name1):
        for j in range(len_name2):
            # If characters at current positions match
            if name1[i] == name2[j]:
                # Update the table by extending the previous common substring
                lcs_table[i + 1][j + 1] = lcs_table[i][j] + 1

                # Update the max_len if this substring is the longest
                max_len = max(max_len, lcs_table[i + 1][j + 1])

    # Determine the maximum possible length
    max_len_name = max(len(name1), len(name2))

    # Return the Longest Common Substring (0 to 1)
    return max_len / max_len_name if max_len_name > 0 else 0


if __name__ == "__main__":
    name1 = "joe biden"
    name2 = "joseph biden"
    name3 = "joseph robinette biden"

    jaccard_index1 = jaccard(name1, name2)
    jaccard_index2 = jaccard(name1, name3)
    jaccard_index3 = jaccard(name2, name3)
    print(f"Jaccard index score between {name1} - {name2} : {jaccard_index1}")
    print(f"Jaccard index score between {name1} - {name3} : {jaccard_index2}")
    print(f"Jaccard index score between {name2} - {name3} : {jaccard_index3}")

    lcsubstr1 = longest_common_substring(name1, name2)
    lcsubstr2 = longest_common_substring(name1, name3)
    lcsubstr3 = longest_common_substring(name2, name3)
    print(f"Longest common substring for {name1} - {name2} : {lcsubstr1}")
    print(f"Longest common substring for {name1} - {name3} : {lcsubstr2}")
    print(f"Longest common substring for {name2} - {name3} : {lcsubstr3}")
