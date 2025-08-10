import pandas as pd
import textdistance as td

# Suffix tokens to ignore when comparing last names
SUFFIXES = {"jr", "sr", "iii", "iv", "ii"}


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
    return len(intersection) / len(union) if union else 0.0


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
    return max_len / max_len_name if max_len_name > 0 else 0.0


def first_name_jaro(name1, name2):
    """Jaro similarity on first tokens."""
    s1 = name1.split()
    s2 = name2.split()
    first1 = s1[0] if s1 else ""
    first2 = s2[0] if s2 else ""
    return td.jaro.normalized_similarity(first1, first2)


def strip_suffix(tokens):
    """Removes common suffixes from the last token."""
    while tokens and tokens[-1].rstrip('.') in SUFFIXES:
        tokens.pop()
    return tokens


def last_name_jaro(name1, name2):
    """Jaro similarity on last substantive token (suffixes removed)."""
    t1 = strip_suffix(name1.split())
    t2 = strip_suffix(name2.split())
    last1 = t1[-1] if t1 else ""
    last2 = t2[-1] if t2 else ""
    return td.jaro.normalized_similarity(last1, last2)


def levenshtein_norm(name1, name2):
    """Normalized Levenshtein similarity: 1 - distance/max_len."""
    a = name1.strip()
    b = name2.strip()
    max_len = max(len(a), len(b), 1)
    dist = td.levenshtein.distance(a, b)
    return 1.0 - dist / max_len


def token_count_diff(name1, name2):
    """Absolute difference in token counts."""
    return abs(len(name1.split()) - len(name2.split()))


def initials_match_ratio(name1, name2):
    """Shared initials ratio over union of initials sets."""
    inits1 = {token[0] for token in name1.split() if token}
    inits2 = {token[0] for token in name2.split() if token}
    union = len(inits1 | inits2)
    if union == 0:
        return 0.0
    return len(inits1 & inits2) / union


def is_initial_token(token):
    """True if token is a single alphabetical letter."""
    return len(token) == 1 and token.isalpha()


def abbreviation_forms(name):
    """
    Extracts ordered initials if 'name' is a valid abbreviation.
    """
    tokens = name.split()

    # Case 1: space-separated initials
    if tokens and all(is_initial_token(t) for t in tokens):
        return tokens

    # Case 2: single fused token
    if len(tokens) == 1:
        token = tokens[0]
        if token.isalpha() and len(token) > 1:
            return list(token)

    return []


def covers(abbrev_name, full_name):
    """True if abbrev_name matches initials of full_name."""
    abbrev_letters = abbreviation_forms(abbrev_name)
    if not abbrev_letters:
        return False

    tokens = full_name.split()
    while tokens and tokens[-1] in SUFFIXES:
        tokens.pop()

    if len(tokens) != len(abbrev_letters):
        return False

    return abbrev_letters == [t[0] for t in tokens]


def initials_full_cover(name1, name2):
    """1 if either name is abbreviation of the other, else 0."""
    return 1 if covers(name1, name2) or covers(name2, name1) else 0


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
    # Intermediate token count diff (not kept as a feature)
    batch["token_count_diff_tmp"] = batch.apply(
        lambda x: token_count_diff(x["name1"], x["name2"]),
        axis=1
    )
    # Compute initials match ratio
    batch["initials_match_ratio"] = batch.apply(
        lambda x: initials_match_ratio(x["name1"], x["name2"]),
        axis=1
    )
    # Compute initials full cover
    batch["initials_full_cover"] = batch.apply(
        lambda x: initials_full_cover(x["name1"], x["name2"]),
        axis=1,
    )

    # extra_middle: 1 if there is an extra middle token while last names
    # are almost identical
    # Threshold for "≈1" last name similarity set to 0.99
    batch["extra_middle"] = (
        (batch["token_count_diff_tmp"] > 0) & (batch["last_name_jaro"] > 0.99)
    ).astype(int)

    # first_name_mismatch: last names similar but first names very dissimilar
    batch["first_name_mismatch"] = (
        (batch["last_name_jaro"] > 0.95) & (batch["first_name_jaro"] < 0.2)
    ).astype(int)

    # same_last_name: exact last substantive token equality (case-insensitive)
    def _same_last(a, b):
        t1 = strip_suffix(a.split())
        t2 = strip_suffix(b.split())
        last1 = t1[-1].lower() if t1 else ""
        last2 = t2[-1].lower() if t2 else ""
        return 1 if last1 and last1 == last2 else 0
    batch["same_last_name"] = batch.apply(
        lambda x: _same_last(x["name1"], x["name2"]), axis=1
    )

    # Drop temporary column
    batch.drop(columns=["token_count_diff_tmp"], inplace=True)

    return batch


def extract_features(input_path, output_path, batch_size=5000):
    """
    Process a large CSV file in batches, compute features, and add them.

    Args:
        input_path (str): Path to the input CSV.
        output_path (str): Path to save the CSV with features.
        batch_size (int): Number of rows to process at a time.
    """
    first_batch = True
    total_processed = 0

    for i, chunk in enumerate(
        pd.read_csv(input_path, chunksize=batch_size, dtype=str)
    ):
        # Compute features for the current batch
        chunk = compute_features(chunk)

        # Save the batch to CSV (overwrite if first, append otherwise)
        chunk.to_csv(
            output_path,
            mode="w" if first_batch else "a",
            header=first_batch,
            index=False,
            encoding="utf-8"
        )

        total_processed += len(chunk)
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
        'initials_match_ratio',
        'initials_full_cover',
        'extra_middle',
        'first_name_mismatch',
        'same_last_name',
    ]

    # Calculate features
    first = first_name_jaro(name1, name2)
    last = last_name_jaro(name1, name2)
    jw = td.jaro_winkler.normalized_similarity(name1, name2)
    lc = longest_common_substring(name1, name2)
    jac = jaccard(name1, name2)
    lev = levenshtein_norm(name1, name2)
    init_ratio = initials_match_ratio(name1, name2)
    init_cover = initials_full_cover(name1, name2)
    tokdiff = token_count_diff(name1, name2)
    extra_middle = 1 if (tokdiff > 0 and last > 0.99) else 0
    first_name_mismatch = 1 if (last > 0.95 and first < 0.2) else 0
    t1 = strip_suffix(name1.split())
    t2 = strip_suffix(name2.split())
    last1 = t1[-1].lower() if t1 else ""
    last2 = t2[-1].lower() if t2 else ""
    same_last_name = 1 if last1 and last1 == last2 else 0
    features = [
        first,
        last,
        jw,
        lc,
        jac,
        lev,
        init_ratio,
        init_cover,
        extra_middle,
        first_name_mismatch,
        same_last_name,
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
