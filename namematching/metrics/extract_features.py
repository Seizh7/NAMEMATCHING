# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import pandas as pd
import textdistance as td

# Suffix tokens to ignore when comparing last names
SUFFIXES = {"jr", "sr", "iii", "iv", "ii"}

CORE_FEATURES = [
    'first_name_jw',
    'last_name_jaro',
    'jaro_winkler',
    'levenshtein_norm',
    'initials_full_cover',
    'first_name_mismatch',
    'first_initial_match',
    'first_initial_expansion',
    'middle_initial_match',
    'acronym_similarity',
    'mixed_acronym_full',
]


def normalize_name(name):
    """Normalize name by removing extra spaces and standardizing dots."""
    return ' '.join(name.strip().split())


def extract_tokens_with_initials(name):
    """
    Extract tokens from name, handling initials with dots.

    Args:
        name (str): Full name string.

    Returns:
        tuple: (tokens, initials_positions) where initials_positions
            marks which tokens are initials.
    """
    name = normalize_name(name)
    tokens = name.split()
    initials_positions = []

    for i, token in enumerate(tokens):
        # Remove dots for analysis
        clean_token = token.rstrip('.')
        # Consider as initial if: single letter,
        # or single letter followed by dot
        if len(clean_token) == 1 and clean_token.isalpha():
            initials_positions.append(i)

    return tokens, initials_positions


def is_initial_token(token):
    """True if token is a single alphabetical letter, with or without dot."""
    clean_token = token.rstrip('.')
    return len(clean_token) == 1 and clean_token.isalpha()


def get_first_letter(token):
    """Get first letter of token, handling dots."""
    clean_token = token.rstrip('.')
    return clean_token[0].lower() if clean_token else ""


def first_initial_match(name1, name2):
    """
    Check if first tokens match by initial.
    Returns 1 if first letters match, 0 otherwise.

    Args:
        name1 (str): First name string.
        name2 (str): Second name string.

    Returns:
        int: 1 if first letters match, 0 otherwise.
    """
    tokens1 = name1.split()
    tokens2 = name2.split()

    if not tokens1 or not tokens2:
        return 0

    first1 = get_first_letter(tokens1[0])
    first2 = get_first_letter(tokens2[0])

    return 1 if first1 == first2 else 0


def first_initial_expansion(name1, name2):
    """
    Check if one name has an initial where the other has a full first name.

    Args:
        name1 (str): First name string.
        name2 (str): Second name string.

    Returns:
        float: Similarity score between 0 and 1.
    """
    tokens1 = name1.split()
    tokens2 = name2.split()

    if not tokens1 or not tokens2:
        return 0

    first1, first2 = tokens1[0], tokens2[0]

    # Case 1: name1 is initial, name2 is full name
    if is_initial_token(first1) and not is_initial_token(first2):
        initial = get_first_letter(first1)
        full_first = get_first_letter(first2)
        return 0.8 if initial == full_first else 0

    # Case 2: name2 is initial, name1 is full name
    if is_initial_token(first2) and not is_initial_token(first1):
        initial = get_first_letter(first2)
        full_first = get_first_letter(first1)
        return 0.8 if initial == full_first else 0

    return 0


def middle_initial_match(name1, name2):
    """
    Check if middle initials match between names.

    Args:
        name1 (str): First name string.
        name2 (str): Second name string.

    Returns:
        float: 1 if all middle initials match, partial score
            for partial matches.
    """
    tokens1, _ = extract_tokens_with_initials(name1)
    tokens2, _ = extract_tokens_with_initials(name2)

    # Get middle tokens (exclude first and last)
    if len(tokens1) <= 2 and len(tokens2) <= 2:
        return 0  # No middle names

    middle1 = tokens1[1:-1] if len(tokens1) > 2 else []
    middle2 = tokens2[1:-1] if len(tokens2) > 2 else []

    if not middle1 or not middle2:
        return 0

    # Extract middle initials
    initials_m1 = [
        get_first_letter(token)
        for token in middle1
        if is_initial_token(token)
    ]
    initials_m2 = [
        get_first_letter(token)
        for token in middle2
        if is_initial_token(token)
    ]

    if not initials_m1 or not initials_m2:
        return 0

    # Count matching initials
    matches = sum(1 for i1 in initials_m1 if i1 in initials_m2)
    total = max(len(initials_m1), len(initials_m2))

    return matches / total if total > 0 else 0


def acronym_similarity(name1, name2):
    """
    Compute similarity when one or both names contain multiple initials.
    Handles cases like 'A. B. Johnson' vs 'Andrew Bob Johnson'.

    Args:
        name1 (str): First name string.
        name2 (str): Second name string.

    Returns:
        float: Similarity score between 0 and 1.
    """
    tokens1, initials_pos1 = extract_tokens_with_initials(name1)
    tokens2, initials_pos2 = extract_tokens_with_initials(name2)

    # If neither name has initials, return 0
    if not initials_pos1 and not initials_pos2:
        return 0

    # Extract initials from each name
    initials1 = [get_first_letter(tokens1[i]) for i in initials_pos1]
    initials2 = [get_first_letter(tokens2[i]) for i in initials_pos2]

    # If one name is all initials, compare with first letters of
    # other name's tokens
    # All but last token are initials
    if len(initials_pos1) == len(tokens1) - 1:
        other_initials = [get_first_letter(token) for token in tokens2[:-1]]
        matches = sum(
            1 for i1, i2 in zip(initials1, other_initials) if i1 == i2
        )
        return (
            matches / max(len(initials1), len(other_initials))
            if initials1 or other_initials else 0
        )
    # All but last token are initials
    if len(initials_pos2) == len(tokens2) - 1:
        other_initials = [get_first_letter(token) for token in tokens1[:-1]]
        matches = sum(
            1 for i1, i2 in zip(initials2, other_initials) if i1 == i2
        )
        return (
            matches / max(len(initials2), len(other_initials))
            if initials2 or other_initials else 0
        )

    # Both have some initials - compare the initials directly
    if initials1 and initials2:
        common = set(initials1) & set(initials2)
        total = set(initials1) | set(initials2)
        return len(common) / len(total) if total else 0

    return 0


def mixed_acronym_full(name1, name2):
    """
    Handle mixed cases like 'A. Mitch McConnell' vs 'Abraham McConnell'.

    Args:
        name1 (str): First name string.
        name2 (str): Second name string.

    Returns:
        float: High score if initials match but some full names are missing.
    """
    tokens1 = name1.split()
    tokens2 = name2.split()

    if len(tokens1) != len(tokens2) and abs(len(tokens1) - len(tokens2)) <= 2:
        # Different number of tokens, might be mixed case

        # Create signature for each name (I=initial, F=full, L=last)
        def get_signature(tokens):
            sig = []
            for i, token in enumerate(tokens):
                if i == len(tokens) - 1:  # Last token
                    sig.append('L')
                elif is_initial_token(token):
                    sig.append('I')
                else:
                    sig.append('F')
            return sig

        sig1 = get_signature(tokens1)
        sig2 = get_signature(tokens2)

        # Check if signatures are compatible
        if 'I' in sig1 or 'I' in sig2:
            # Compare last names first
            last_sim = last_name_jaro(name1, name2)
            if last_sim > 0.9:
                # Compare first initials
                first_match = first_initial_match(name1, name2)
                return 0.7 * first_match + 0.3 * last_sim

    return 0


def strip_suffix(tokens):
    """Remove common suffixes from the last token."""
    while tokens and tokens[-1].rstrip('.').lower() in SUFFIXES:
        tokens.pop()
    return tokens


def last_name_jaro(name1, name2):
    """Jaro similarity on last substantive token (suffixes removed)."""
    token1 = strip_suffix(name1.split())
    token2 = strip_suffix(name2.split())
    last1 = token1[-1] if token1 else ""
    last2 = token2[-1] if token2 else ""
    return td.jaro.normalized_similarity(last1, last2)


def first_name_jw(name1, name2):
    """Jaro-Winkler similarity on first tokens (prefix sensitive)."""
    name1 = name1.split()
    name2 = name2.split()
    first1 = name1[0] if name1 else ""
    first2 = name2[0] if name2 else ""
    return td.jaro_winkler.normalized_similarity(first1, first2)


def levenshtein_norm(name1, name2):
    """Normalized Levenshtein similarity: 1 - distance/max_len."""
    name1 = name1.strip()
    name2 = name2.strip()
    max_len = max(len(name1), len(name2), 1)
    dist = td.levenshtein.distance(name1, name2)
    return 1.0 - dist / max_len


def abbreviation_forms(name):
    """
    Extracts ordered initials if 'name' is a valid abbreviation.

    Args:
        name (str): The name string to extract initials from.

    Returns:
        list: A list of ordered initials if 'name' is a valid abbreviation,
            else an empty list.
    """
    tokens = name.split()

    # Case 1: space-separated initials
    if tokens and all(is_initial_token(t) for t in tokens):
        return tokens

    # Case 2: single fused token (alphabetic, length > 1)
    if len(tokens) == 1:
        token = tokens[0]
        if token.isalpha() and len(token) > 1:
            return list(token)

    return []


def covers(abbrev_name, full_name):
    """
    True if abbrev_name matches initials of full_name.

    Args:
        abbrev_name (str): The abbreviated name to check.
        full_name (str): The full name to match against.

    Returns:
        bool: True if abbrev_name matches initials of full_name, else False.
    """
    abbrev_letters = abbreviation_forms(abbrev_name)
    if not abbrev_letters:
        return False

    tokens = full_name.split()
    # remove suffixes from the tail
    tokens = strip_suffix(tokens)

    if len(tokens) != len(abbrev_letters):
        return False

    return abbrev_letters == [t[0] for t in tokens]


def initials_full_cover(name1, name2):
    """1 if either name is abbreviation of the other, else 0."""
    return 1 if covers(name1, name2) or covers(name2, name1) else 0


def first_name_mismatch_flag(name1, name2, last_thresh=0.95, first_thresh=0.2):
    """
    Binary flag when last names are very similar but first names
    are very dissimilar.

    Args:
        name1 (str): The first name string.
        name2 (str): The second name string.
        last_thresh (float): Threshold for last name similarity.
        first_thresh (float): Threshold for first name similarity.

    Returns:
        int: 1 if last names are similar but first names are dissimilar,
            else 0.
    """
    return int(last_name_jaro(name1, name2) >= last_thresh and
               first_name_jw(name1, name2) < first_thresh)


def compute_features(batch):
    """
    Compute a set of string similarity and rule-based features for a batch
    of name pairs.

    Args:
        batch (pd.DataFrame):
            A chunk of the dataset containing at least the columns:
            - "name1" (str): First name string in the pair
            - "name2" (str): Second name string in the pair

    Returns:
        pd.DataFrame:
            The same batch with additional numeric feature columns describing
            different aspects of similarity between name1 and name2.
    """
    # Ensure names are lowercase for consistent comparisons
    batch["name1"] = batch["name1"].str.lower()
    batch["name2"] = batch["name2"].str.lower()

    # --- String similarity metrics ---
    # Jaro-Winkler similarity computed only on the first tokens (first names),
    # more sensitive to prefix matches than plain Jaro.
    batch["first_name_jw"] = batch.apply(
        lambda x: first_name_jw(x["name1"], x["name2"]),
        axis=1
    )

    # Jaro similarity computed only on the last tokens (last names),
    # after removing common suffixes like Jr, Sr, III.
    batch["last_name_jaro"] = batch.apply(
        lambda x: last_name_jaro(x["name1"], x["name2"]),
        axis=1
    )

    # Full-string Jaro-Winkler similarity: captures general similarity of
    # the complete names, giving extra weight to common prefixes.
    batch["jaro_winkler"] = batch.apply(
        lambda x: td.jaro_winkler.normalized_similarity(
            x["name1"], x["name2"]
        ),
        axis=1
    )

    # Normalized Levenshtein similarity: 1 - (edit distance / max length).
    # Captures overall character-level edits required to transform one name
    # into the other.
    batch["levenshtein_norm"] = batch.apply(
        lambda x: levenshtein_norm(x["name1"], x["name2"]),
        axis=1
    )

    # Initials-based coverage
    # 1 if one name is an abbreviation (ordered initials) of the other,
    # else 0.
    batch["initials_full_cover"] = batch.apply(
        lambda x: initials_full_cover(x["name1"], x["name2"]),
        axis=1
    )

    # Rule-based mismatch flags
    # Flags possible homonyms: last names are very similar, but first names
    # are dissimilar.
    batch["first_name_mismatch"] = batch.apply(
        lambda x: first_name_mismatch_flag(x["name1"], x["name2"]),
        axis=1
    )

    # 1 if the first initials match, else 0.
    batch["first_initial_match"] = batch.apply(
        lambda x: first_initial_match(x["name1"], x["name2"]),
        axis=1
    )

    # 1 if the first name in one string is just an initial and matches the
    # first letter of the other full first name.
    batch["first_initial_expansion"] = batch.apply(
        lambda x: first_initial_expansion(x["name1"], x["name2"]),
        axis=1
    )

    # 1 if middle initials (when present) match, else 0.
    batch["middle_initial_match"] = batch.apply(
        lambda x: middle_initial_match(x["name1"], x["name2"]),
        axis=1
    )

    # Similarity score between acronyms formed from each name’s tokens.
    # Useful when both names are abbreviations.
    batch["acronym_similarity"] = batch.apply(
        lambda x: acronym_similarity(x["name1"], x["name2"]),
        axis=1
    )

    # 1 if one name is an acronym and the other is a full expansion matching
    # the acronym letters.
    batch["mixed_acronym_full"] = batch.apply(
        lambda x: mixed_acronym_full(x["name1"], x["name2"]),
        axis=1
    )

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
    name1 = (name1 or "").lower()
    name2 = (name2 or "").lower()

    feature_names = CORE_FEATURES

    first = first_name_jw(name1, name2)
    last = last_name_jaro(name1, name2)
    jw_full = td.jaro_winkler.normalized_similarity(name1, name2)
    lev = levenshtein_norm(name1, name2)
    init_cover = initials_full_cover(name1, name2)
    fname_mismatch = first_name_mismatch_flag(name1, name2)
    first_init_match = first_initial_match(name1, name2)
    first_init_expansion = first_initial_expansion(name1, name2)
    middle_init_match = middle_initial_match(name1, name2)
    acronym_sim = acronym_similarity(name1, name2)
    mixed_acr_full = mixed_acronym_full(name1, name2)

    features = [
        first,
        last,
        jw_full,
        lev,
        init_cover,
        fname_mismatch,
        first_init_match,
        first_init_expansion,
        middle_init_match,
        acronym_sim,
        mixed_acr_full,
    ]

    return pd.DataFrame([features], columns=feature_names)


if __name__ == "__main__":
    data = [
        {"name1": "joe biden", "name2": "joseph biden"},
        {"name1": "joe biden", "name2": "joseph robinette biden"},
        {"name1": "joseph biden", "name2": "joseph robinette biden"},
        {"name1": "barack obama", "name2": "barack h obama"},
        {"name1": "donald trump", "name2": "donald j trump"},
        {"name1": "joe biden", "name2": "barack obama"},
        {"name1": "a. mitch mcconnell", "name2": "abraham mcconnell"},
        {"name1": "j. k. rowling", "name2": "joanne kathleen rowling"},
        {"name1": "f. scott fitzgerald", "name2": "francis fitzgerald"},
        {"name1": "a. lincoln", "name2": "abraham lincoln"},
        {"name1": "r. j. smith", "name2": "robert john smith"},
    ]

    df = pd.DataFrame(data)
    feats = compute_features(df)
    print(feats)

    # Test single extraction
    single = extract_individual_features(
        "a. mitch mcconnell",
        "abraham mcconnell"
    )
    print(
        "\nSingle feature extraction for 'a. mitch mcconnell' "
        "vs 'abraham mcconnell':"
    )
    print(single)
