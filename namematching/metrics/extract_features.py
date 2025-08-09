import pandas as pd
import textdistance as td


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


def first_name_jaro(name1, name2):
    """Jaro similarity on first tokens."""
    first1 = name1.split()[0].lower() if name1.split() else ""
    first2 = name2.split()[0].lower() if name2.split() else ""
    return td.jaro.normalized_similarity(first1, first2)


def last_name_jaro(name1, name2):
    """Jaro similarity on last tokens."""
    last1 = name1.split()[-1].lower() if name1.split() else ""
    last2 = name2.split()[-1].lower() if name2.split() else ""
    return td.jaro.normalized_similarity(last1, last2)


def levenshtein_norm(name1: str, name2: str) -> float:
    """Normalized Levenshtein similarity: 1 - distance/max_len."""
    a = name1.lower().strip()
    b = name2.lower().strip()
    max_len = max(len(a), len(b), 1)
    dist = td.levenshtein.distance(a, b)
    return 1 - dist / max_len


def token_count_diff(name1: str, name2: str) -> int:
    """Absolute difference in token counts."""
    return abs(len(name1.split()) - len(name2.split()))


def initials_match_ratio(name1: str, name2: str) -> float:
    """Shared initials ratio over union of initials sets."""
    inits1 = {tok[0].lower() for tok in name1.split() if tok}
    inits2 = {tok[0].lower() for tok in name2.split() if tok}
    union = len(inits1 | inits2)
    if union == 0:
        return 0.0
    return len(inits1 & inits2) / union


def compute_features(batch):
    """
    Compute similarity features on a batch of name pairs.

    Args:
        batch (pd.DataFrame): A chunk of the full CSV.

    Returns:
        pd.DataFrame: The same batch with new feature columns added.
    """
    batch["first_name_jaro"] = batch.apply(
        lambda x: first_name_jaro(x["name1"], x["name2"]),
        axis=1
    )
    batch["last_name_jaro"] = batch.apply(
        lambda x: last_name_jaro(x["name1"], x["name2"]),
        axis=1
    )
    # Compute Jaro-Winkler similarity
    batch["jaro_winkler"] = batch.apply(
        lambda x: td.jaro_winkler.normalized_similarity(
            x["name1"], x["name2"]
        ),
        axis=1
    )
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
    # Compute Levenshtein similarity
    batch["levenshtein_norm"] = batch.apply(
        lambda x: levenshtein_norm(x["name1"], x["name2"]),
        axis=1
    )
    # Compute token count difference
    batch["token_count_diff"] = batch.apply(
        lambda x: token_count_diff(x["name1"], x["name2"]),
        axis=1
    )
    # Compute initials match ratio
    batch["initials_match_ratio"] = batch.apply(
        lambda x: initials_match_ratio(x["name1"], x["name2"]),
        axis=1
    )

    return batch


def extract_features(input_path, output_path, batch=5000):
    """
    Process a large CSV file in batches, compute features, and add them.

    Args:
        input_path (str): Path to the input CSV.
        output_path (str): Path to save the CSV with features.
        batch_size (int): Number of rows to process at a time.
    """
    first_batch = True
    total_processed = 0

    for i, batch in enumerate(pd.read_csv(input_path, chunksize=batch)):
        # Compute features for the current batch
        batch = compute_features(batch)

        # Save the batch to CSV (overwrite if first, append otherwise)
        batch.to_csv(
            output_path,
            mode="w" if first_batch else "a",
            header=first_batch,
            index=False,
            encoding="utf-8"
        )

        total_processed += len(batch)
        print(f"Batch {i + 1} saved ({total_processed} rows processed)")
        first_batch = False


def extract_individual_features(name1, name2):
    """
    Extracts features for a single pair of names.

    Args:
        name1 (str): First name
        name2 (str): Second name

    Returns:
        pd.DataFrame: Features in a format compatible with the trained scaler
    """
    # Define feature names in the same order as during training
    feature_names = [
        'first_name_jaro',
        'last_name_jaro',
        'jaro_winkler',
        'lcsubstr',
        'jaccard',
        'levenshtein_norm',
        'token_count_diff',
        'initials_match_ratio'
    ]

    # Calculate features
    features = [
        first_name_jaro(name1, name2),
        last_name_jaro(name1, name2),
        td.jaro_winkler.normalized_similarity(name1, name2),
        longest_common_substring(name1, name2),
        jaccard(name1, name2),
        levenshtein_norm(name1, name2),
        token_count_diff(name1, name2),
        initials_match_ratio(name1, name2),
    ]

    # Return as DataFrame with correct feature names
    return pd.DataFrame([features], columns=feature_names)


if __name__ == "__main__":
    name1 = "joe biden"
    name2 = "joseph biden"
    name3 = "joseph robinette biden"

    # Startswith same
    start1 = first_name_jaro(name1, name2)
    start2 = first_name_jaro(name1, name3)
    start3 = first_name_jaro(name2, name3)
    print(f"Jaro first name {name1} - {name2} : {start1}")
    print(f"Jaro first name {name1} - {name3} : {start2}")
    print(f"Jaro first name {name2} - {name3} : {start3}")

    # Endswith same
    end1 = last_name_jaro(name1, name2)
    end2 = last_name_jaro(name1, name3)
    end3 = last_name_jaro(name2, name3)
    print(f"Jaro last name {name1} - {name2} : {end1}")
    print(f"Jaro last name {name1} - {name3} : {end2}")
    print(f"Jaro last name {name2} - {name3} : {end3}")

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

    features_np = extract_individual_features(name1, name2)
    print(f"Individual features between {name1} and {name2}: {features_np}")
