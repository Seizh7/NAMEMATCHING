# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from namematching.config import CONFIG

# Define characters
CHARACTERS = "abcdefghijklmnopqrstuvwxyz "
MAX_LEN = 40  # Fixed maximum length for character-level encoding

# Expected feature columns
FEATURE_COLUMNS = [
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

# Mapping from character to integer
char_to_idx = {c: i + 1 for i, c in enumerate(CHARACTERS)}


def char_tokenizer(name):
    """
    Converts a name string into a fixed-length array of character indices.

    Args:
        name (str): The name to encode.

    Returns:
        np.array: An array of length MAX_LEN with character indices.
    """
    name = name.lower()
    name = ''.join(c for c in name if c in CHARACTERS)  # Only allowed chars
    token = [char_to_idx.get(c, 0) for c in name]  # Convert to indices

    # Truncate the token if it's longer than MAX_LEN
    truncated_token = token[:MAX_LEN]

    # Computes how many zeros are needed to pad the sequence
    padding_length = MAX_LEN - len(truncated_token)

    # Combine the token with the padding
    padded_token = truncated_token + [0] * padding_length

    return np.array(padded_token)


def load_and_prepare_data(csv_path):
    """
    Loads a dataset from CSV, encodes character-level inputs and features,
    splits into train/test sets, and returns inputs and labels.

    Args:
        csv_path (str): Path to CSV.

    Returns:
        tuple: ((X1_train, X2_train, F_train, y_train),
                (X1_test, X2_test, F_test, y_test),
                scaler, char_to_idx)
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Encode both name columns using the character tokenizer
    X_name1 = np.stack(df["name1"].map(char_tokenizer))
    X_name2 = np.stack(df["name2"].map(char_tokenizer))

    # Ensure all expected columns exist
    required = ['name1', 'name2', 'label'] + FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # Encode names to fixed-length char indices
    X_name1 = np.stack(df['name1'].map(char_tokenizer))
    X_name2 = np.stack(df['name2'].map(char_tokenizer))

    # Scale features
    scaler = StandardScaler()
    X_feats = scaler.fit_transform(df[FEATURE_COLUMNS])
    y = df['label'].values

    (X1_train, X1_test,
     X2_train, X2_test,
     F_train, F_test,
     y_train, y_test) = train_test_split(
        X_name1, X_name2, X_feats, y, test_size=0.2, random_state=42
    )

    return (
        (X1_train, X2_train, F_train, y_train),
        (X1_test, X2_test, F_test, y_test),
        scaler,
        char_to_idx,
    )


if __name__ == "__main__":
    name = "joe biden"
    token = char_tokenizer(name)

    print(char_to_idx)
    print(f"Nom : {name}")
    print(f"Token : {name}")

    data_path = CONFIG.data_dir / "test_data.csv"
    (
        (X1_train, X2_train, F_train, y_train),
        (X1_test, X2_test, F_test, y_test),
        scaler,
        mapping,
    ) = load_and_prepare_data(data_path)

    print("X1_train shape:", X1_train.shape)
    print("X2_train shape:", X2_train.shape)
    print("F_train shape:", F_train.shape)
    print("y_train:", y_train)
    print("X1_test shape:", X1_test.shape)
