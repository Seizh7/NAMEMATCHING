from itertools import combinations, cycle
import utils
import random
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
    # If no positive pairs, don't generate negatives (avoid imbalance noise)
    if n_positives == 0:
        return []

    # Get all QIDs except the current one
    all_qids = list(data.keys())
    other_qids = [other for other in all_qids if other != qid]

    names1_list = [data[qid]["name"]] + data[qid].get("aliases", [])
    names1_cycle = cycle(names1_list)

    # Determine how many negatives to generate
    n_negatives = max(n_positives, len(names1_list))
    if n_negatives == 0:
        n_negatives = 1  # ensure at least one negative pair

    n_negatives = min(n_negatives, len(other_qids))

    # Randomly sample other people to pair against
    selected_qids = random.sample(other_qids, k=n_negatives)

    negative_pairs = []

    for other_qid in selected_qids:
        name2 = data[other_qid].get("name", "").strip()
        if not name2:
            continue  # skip if name2 is missing or empty

        # Create negatives pairs
        for _ in range(len(names1_list)):  # prevent infinite loop
            name1 = next(names1_cycle)
            if name1 != name2:
                negative_pairs.append({
                    "name1": name1,
                    "name2": name2,
                    "label": 0
                })

                # Build negative with firstname of name2 +
                # lastname of name1
                name1_parts = name1.split()
                name2_parts = name2.split()
                if len(name1_parts) >= 2 and len(name2_parts) >= 2:
                    fake_name2 = f"{name2_parts[0]} {name1_parts[-1]}"
                    if name1 != fake_name2:
                        negative_pairs.append({
                            "name1": name1,
                            "name2": fake_name2,
                            "label": 0
                        })
                break  # move to the next other_qid

    return negative_pairs


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
    name = data["Q109463"]["name"]
    aliases = data["Q109463"]["aliases"]
    pos = generate_positive_pairs("Q109463", data)
    print(f"positive pair(s): {pos}")

    neg = generate_negative_pairs("Q109463", data, len(pos))
    print(f"negative pair(s): {neg}")
