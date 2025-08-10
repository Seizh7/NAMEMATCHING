from itertools import combinations
import utils
import pandas as pd
import os
import csv


def generate_positive_pairs(qid, data):
    """
    Generates all positive pairs for a single person,
    based on their primary name and known aliases.

    Args:
        qid (str): The Wikidata QID of the person.
        data (dict): Dictionary of people with their names and aliases.

    Returns:
        list[dict]: List of dictionaries representing name pairs with label 1
    """
    primary_name = data[qid]["name"]
    aliases = data[qid]["aliases"]

    # Combine all known names and remove duplicates
    all_names = list(set([primary_name] + aliases))

    # Generate all unique combinations of two names from the set
    name_pairs = combinations(all_names, 2)

    # Create labeled entries for each name pair
    labeled_pairs = []
    for name1, name2 in name_pairs:
        labeled_pairs.append({
            "name1": name1,
            "name2": name2,
            "label": 1
        })

    return labeled_pairs


def generate_negative_pairs(qid, data, n_positives):
    """
    Generates negative name pairs (label = 0) for a given person,
    by pairing their name (or aliases) with names from other individuals.

    Args:
        qid (str): The Wikidata QID of the person.
        data (dict): Dictionary of people with their names and aliases.
        n_positives (int): Number of positive pairs generated for this person.

    Returns:
        list[dict]: A list of negative name pairs.
    """
    # Get all QIDs except the current one
    all_qids = list(data.keys())
    other_qids = [other for other in all_qids if other != qid]

    names1_list = [data[qid]["name"]] + data[qid].get("aliases", [])

    # Target counts: 3 negatives per positive, half of them "hard"
    total_needed = n_positives * 3
    hard_needed = int(total_needed * 0.5)

    negatives = []
    seen = set()  # to avoid duplicate pairs
    hard_count = 0

    i = 0
    # Suffixes we avoid using as a family name in hard negatives
    SUFFIXES = {"jr", "sr", "iii", "iv", "ii"}

    def last_non_suffix(parts):
        """Return last token from parts that's not a suffix, else None.

        We strip simple punctuation (comma / period) at the end for the check,
        but keep the original token when reconstructing the fake name.
        """
        for token in reversed(parts):
            if token.lower() not in SUFFIXES:
                return token
        return None

    while len(negatives) < total_needed:
        # Pick a source name (cycles through names1_list)
        name1 = names1_list[i % len(names1_list)]

        # Pick a target "other" person and use their main name
        other_person = data[other_qids[i % len(other_qids)]]
        name2 = other_person.get("name", "").strip()
        i += 1

        if not name2 or name1 == name2:
            continue
        if (name1, name2) in seen:
            continue

        # Add a standard negative pair
        negatives.append({"name1": name1, "name2": name2, "label": 0})
        seen.add((name1, name2))

        # Optionally add a "hard negative" (same-family-name construction)
        if hard_count < hard_needed and len(negatives) < total_needed:
            parts1 = name1.split()
            parts2 = name2.split()
            if len(parts1) >= 2 and len(parts2) >= 2:
                surname = last_non_suffix(parts1)
                if surname and surname.lower() not in SUFFIXES:
                    fake_name2 = f"{parts2[0]} {surname}".strip()
                    # Skip if last token of fake name is still a suffix
                    last_token_clean = fake_name2.split()[-1].lower()
                    if last_token_clean in SUFFIXES:
                        fake_name2 = None
                    if (
                        fake_name2 and
                        fake_name2 not in (name1, name2) and
                        (name1, fake_name2) not in seen
                    ):
                        negatives.append(
                            {"name1": name1, "name2": fake_name2, "label": 0}
                        )
                        seen.add((name1, fake_name2))

            if len(parts1) >= 1 and len(parts2) >= 2:
                firstname = parts1[0]
                surname2 = last_non_suffix(parts2)
                # Build a fake name that keeps the same first name
                # but swaps the last name
                if surname2 and surname2.lower() not in SUFFIXES:
                    fake_name2 = f"{firstname} {surname2}".strip()
                    # Avoid using the exact original names or duplicates
                    if (
                        fake_name2 not in (name1, name2)
                        and (name1, fake_name2) not in seen
                    ):
                        negatives.append(
                            {
                                "name1": name1,
                                "name2": fake_name2,
                                "label": 0,
                            }
                        )
                        seen.add((name1, fake_name2))

            hard_count += 1

    # Trim if there are too many pairs
    return negatives[:total_needed]


def generate_pairs(qid, data):
    """
    Generates both positive and negative pairs for a given person.

    Args:
        qid (str): The QID of the person.
        person (dict): A dict with keys "name" and "aliases".
        data (dict): Full dataset.
        n_negatives (int): Number of negative pairs per person.

    Returns:
        list[dict]: A list of pair dictionaries (positive + negative).
    """
    positive = generate_positive_pairs(qid, data)
    negative = generate_negative_pairs(qid, data, len(positive))
    return positive + negative


def init_csv(path):
    """
    Initializes the output CSV file with the appropriate header.

    Args:
        path (str): Path to the CSV file.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["name1", "name2", "label"])
        writer.writeheader()


def write_batch_to_csv(batch, output_path):
    """
    Writes a batch of name pairs to the CSV file.

    Args:
        batch (list): List of dictionaries.
        output_path (str): Path to the output file.

    Returns:
        None
    """
    df = pd.DataFrame(batch)

    df["name1"] = df["name1"].str.lower()
    df["name2"] = df["name2"].str.lower()

    df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=False,
        encoding="utf-8"
    )


def generate_and_save_pairs(
    data,
    output_path,
    batch_size=50,
):
    """
    Generates name pairs for all individuals in the dataset and writes them to
    CSV in batches.

    Args:
        data (dict): Full dataset
        output_path (str): Output CSV path.
        batch_size (int): Number of individuals per batch.
        n_negatives (int): Number of negative pairs to generate per individual.

    Returns:
        None
    """
    init_csv(output_path)

    batch = []
    qids = list(data.keys())

    for i, qid in enumerate(qids):
        # Skip persons without aliases
        aliases = data[qid].get("aliases", []) or []
        if len(aliases) == 0:
            continue

        pairs = generate_pairs(qid, data)
        batch.extend(pairs)

        # Write every `batch_size` or at the end
        if (i + 1) % batch_size == 0 or (i + 1) == len(qids):
            print(f"Batch {i // batch_size + 1} saved ({(i + 1)}/{len(qids)})")
            write_batch_to_csv(batch, output_path)
            batch = []


if __name__ == "__main__":
    data = utils.load_json("data/test.json")
    name = data["Q122841"]["name"]
    aliases = data["Q122841"]["aliases"]
    pos = generate_positive_pairs("Q122841", data)
    print(f"positive pair(s): {pos}")

    neg = generate_negative_pairs("Q122841", data, len(pos))
    print(f"negative pair(s): {neg}")
