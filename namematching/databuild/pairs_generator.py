from itertools import combinations
import utils


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


if __name__ == "__main__":
    data = utils.load_json("data/test.json")
    name = data["Q109463"]["name"]
    aliases = data["Q109463"]["aliases"]
    pos = generate_positive_pairs("Q109463", data)
    print(f"positive pair(s): {pos}")
