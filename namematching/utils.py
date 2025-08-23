import json
import re
import os


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_name(name: str) -> str:
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

    if not isinstance(name, str):
        name = str(name)

    # Replace some separators with space
    name = re.compile(r"[\-/_]").sub(" ", name)

    # Remove primary punctuation
    name = re.compile(r"[.,'\"]").sub("", name)

    # Replace parentheses with space (keep inner content)
    name = re.sub(r"[()]", " ", name)

    # Collapse whitespace
    name = re.compile(r"\s+").sub(" ", name).strip()

    if not name:
        return ""

    return name.strip()
