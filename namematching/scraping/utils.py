import re


def normalize_name(name):
    """
    Normalizes a personal name by removing punctuation, converting to
    lowercase, and collapsing extra spaces.

    Args:
        name (str): The name to normalize.

    Returns:
        str: Normalized version of the name.
    """
    if not name:
        return ""
    name = name.lower()                  # Lowercase for consistent comparison
    name = re.sub(r"[.']", "", name)     # Remove dots and apostrophes
    name = re.sub(r"[-]", " ", name)     # Replace hyphens with space
    name = re.sub(r"\s+", " ", name)     # Collapse multiple spaces into one
    name = name.strip(", ")              # Remove commas and spaces
    return name.strip()
