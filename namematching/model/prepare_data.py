import numpy as np

# Define characters
CHARACTERS = "abcdefghijklmnopqrstuvwxyz "
MAX_LEN = 40  # Fixed maximum length for character-level encoding

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


if __name__ == "__main__":
    name = "joe biden"
    token = char_tokenizer(name)

    print(char_to_idx)
    print(f"Nom : {name}")
    print(f"Token : {name}")
