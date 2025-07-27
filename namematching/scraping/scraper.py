import requests


def get_name_and_aliases(entity_id):
    """
    Retrieves the English label and aliases of a Wikidata entity.

    Args:
        entity_id (str): The Wikidata entity

    Returns:
        dict: A dictionary containing the entity's name and a list of aliases.
              {
                  "name": <label>,
                  "aliases": [<alias_1>, <alias_2>, ...]
              }
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    
    # Send HTTP GET request
    response = requests.get(url)
    response.raise_for_status()  # If non-200 responses

    data = response.json()

    # Entity's data
    entity = data["entities"][entity_id]

    # Extract the English label (main name)
    label = entity.get("labels", {}).get("en", {}).get("value", "")

    # Extract English aliases
    aliases = entity.get("aliases", {}).get("en", [])
    alias_list = [alias["value"] for alias in aliases]

    return {
        "name": label,
        "aliases": alias_list
    }


if __name__ == "__main__":
    # Example
    result = get_name_and_aliases("Q355522")
    print(result)
