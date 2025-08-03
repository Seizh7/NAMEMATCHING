import pandas as pd


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


def startswith_same(name1, name2):
    """
    Returns 1 if both names start with the same word, else 0.
    """
    return int(name1.split()[0] == name2.split()[0])


def endswith_same(name1, name2):
    """
    Returns 1 if both names end with the same word, else 0.
    """
    return int(name1.split()[-1] == name2.split()[-1])


def compute_features(batch):
    """
    Compute similarity features on a batch of name pairs.

    Args:
        batch (pd.DataFrame): A chunk of the full CSV.

    Returns:
        pd.DataFrame: The same batch with new feature columns added.
    """
    # Compute Jaro similarity (character-level)
    batch["jaro"] = batch.apply(
        lambda x: td.jaro.normalized_similarity(x["name1"], x["name2"]),
        axis=1
    )

    # Compute Jaro-Winkler similarity
    batch["jaro_winkler"] = batch.apply(
        lambda x: td.jaro_winkler.normalized_similarity(
            x["name1"], x["name2"]
        ),
        axis=1
    )

    # Compute Fuzzy ratio with error tracing
    fuzz_scores = []
    for i, row in batch.iterrows():
        try:
            name1 = str(row["name1"])
            name2 = str(row["name2"])
            score = fuzz.ratio(name1, name2) / 100
        except Exception as e:
            print(f"[fuzz] Exception at row {i}: {e}")
            score = 0.0
        fuzz_scores.append(score)

    batch["fuzz_ratio"] = fuzz_scores

    # Compute longest common substring similarity
    batch["lcsubstr"] = batch.apply(
        lambda x: longest_common_substring(x["name1"], x["name2"]),
        axis=1
    )

    # Compute Jaccard similarity
    batch["jaccard"] = batch.apply(
        lambda x: jaccard(x["name1"], x["name2"]),
        axis=1
    )

    # Same starting word
    batch["startswith_same"] = batch.apply(
        lambda x: startswith_same(x["name1"], x["name2"]),
        axis=1
    )

    # Same ending word
    batch["endswith_same"] = batch.apply(
        lambda x: endswith_same(x["name1"], x["name2"]),
        axis=1
    )

    return batch


if __name__ == "__main__":
    name1 = "joe biden"
    name2 = "joseph biden"
    name3 = "joseph robinette biden"

    # Jaccard index
    jaccard_index1 = jaccard(name1, name2)
    jaccard_index2 = jaccard(name1, name3)
    jaccard_index3 = jaccard(name2, name3)
    print(f"Jaccard index score between {name1} - {name2} : {jaccard_index1}")
    print(f"Jaccard index score between {name1} - {name3} : {jaccard_index2}")
    print(f"Jaccard index score between {name2} - {name3} : {jaccard_index3}")

    # Longest common substring
    lcsubstr1 = longest_common_substring(name1, name2)
    lcsubstr2 = longest_common_substring(name1, name3)
    lcsubstr3 = longest_common_substring(name2, name3)
    print(f"Longest common substring for {name1} - {name2} : {lcsubstr1}")
    print(f"Longest common substring for {name1} - {name3} : {lcsubstr2}")
    print(f"Longest common substring for {name2} - {name3} : {lcsubstr3}")

    # Startswith same
    start1 = startswith_same(name1, name2)
    start2 = startswith_same(name1, name3)
    start3 = startswith_same(name2, name3)
    print(f"Startswith same for {name1} - {name2} : {start1}")
    print(f"Startswith same for {name1} - {name3} : {start2}")
    print(f"Startswith same for {name2} - {name3} : {start3}")

    # Endswith same
    end1 = endswith_same(name1, name2)
    end2 = endswith_same(name1, name3)
    end3 = endswith_same(name2, name3)
    print(f"Endswith same for {name1} - {name2} : {end1}")
    print(f"Endswith same for {name1} - {name3} : {end2}")
    print(f"Endswith same for {name2} - {name3} : {end3}")

    data = [
        {"name1": "joe biden", "name2": "joseph biden"},
        {"name1": "joe biden", "name2": "joseph robinette biden"},
        {"name1": "joseph biden", "name2": "joseph robinette biden"},
        {"name1": "barack obama", "name2": "barack h obama"},
        {"name1": "donald trump", "name2": "donald j trump"},
        {"name1": "joe biden", "name2": "barack obama"},
    ]

    test_df = pd.DataFrame(data)
    features_df = compute_features(test_df)

    print(f"{features_df}")
