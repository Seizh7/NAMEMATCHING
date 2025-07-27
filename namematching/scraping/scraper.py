import requests
from SPARQLWrapper import SPARQLWrapper, JSON


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


def get_all_us_legislator_qids():
    """
    Retrieves the Wikidata QIDs of all current or US legislators

    Returns:
        list: A list of QID strings
    """
    # Initialize the SPARQL endpoint for Wikidata
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Define the SPARQL query:
    # - Select all persons (?person) who held positions:
    #   Q4416090 = Member of the United States House of Representatives
    #   Q13218630 = United States Senator
    sparql.setQuery("""
    SELECT DISTINCT ?person WHERE {
      VALUES ?position { wd:Q4416090 wd:Q13218630 }
      ?person wdt:P39 ?position .
    }
    """)

    # Query and parse the results in JSON
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Extract the QID from the full URI
    bindings = results["results"]["bindings"]
    qids = [result["person"]["value"].split("/")[-1] for result in bindings]
    return qids


if __name__ == "__main__":
    # Example
    result = get_name_and_aliases("Q355522")
    print(result)
    print(get_all_us_legislator_qids())
