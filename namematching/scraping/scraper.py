import requests
import time
import utils
from SPARQLWrapper import SPARQLWrapper, JSON

# terms to exclude
EXCLUSION_TERMS = [
    "president",
    "secretary",
    "governor",
    "gov",
    "mayor",
    "senator",
    "sen",
    "congressman",
    "general",
    "colonel",
    "brigadier general",
    "vp",
    "potus",
    "vp potus",
    "rep",
    "first lady",
]


def excluded_term(name):
    """
    Check if the name contains any excluded terms.
    """
    name = " " + name.lower() + " "
    for term in EXCLUSION_TERMS:
        if f" {term} " in name:
            return True
    return False


def get_name_and_aliases(entity_id, langs=["en", "fr", "de"]):
    """
    Retrieves the English label and aliases of a Wikidata entity.

    Args:
        entity_id (str): The Wikidata entity
        langs (str): Choosen language for the label

    Returns:
        dict: A dictionary containing the entity's name and a list of aliases.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    # Send HTTP GET request
    response = requests.get(url)
    response.raise_for_status()  # If non-200 responses

    data = response.json()

    # Entity's data
    entity = data["entities"][entity_id]

    # Extract the label (main name)
    label = ""
    for lang in langs:
        if lang in entity.get("labels", {}):
            label = entity["labels"][lang]["value"]
            break

    # Fallback to enwiki title when no label is available
    if not label:
        sitelinks = entity.get("sitelinks", {})
        if "enwiki" in sitelinks:
            label = sitelinks["enwiki"].get("title", "")

    # Extract English aliases
    aliases = entity.get("aliases", {}).get("en", [])
    alias_list = [alias["value"] for alias in aliases]

    # Normalize name and aliases
    normalized_label = utils.normalize_name(label)
    normalized_aliases = [utils.normalize_name(alias) for alias in alias_list]

    # If the main name is excluded, ignore the entity
    if excluded_term(normalized_label):
        return None

    # Filter aliases containing excluded terms
    filtered_aliases = [
        a for a in normalized_aliases if not excluded_term(a)
    ]

    return {"name": normalized_label, "aliases": filtered_aliases}


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


def process_batch(batch, data):
    """
    Processes a batch of QIDs by retrieving their names and aliases.

    Args:
        batch (list): List of Wikidata QIDs to process.
        data (dict): Existing data dictionary to update.

    Returns:
        dict: Updated data dictionary with new QID entries.
    """
    for qid in batch:
        if qid in data:
            continue  # Skip already processed QIDs

        info = get_name_and_aliases(qid)
        if info:
            data[qid] = info

        time.sleep(0.5)  # Respect Wikidata rate limits

    return data


def get_all_names_with_aliases(qids, output_path, batch_size=10):
    """
    Retrieves names and aliases for a list of QIDs in batches,
    saving progress incrementally to a JSON file.

    Args:
        qids (list): List of Wikidata QIDs to process.
        output_path (str): Path to the output JSON file.
        batch_size (int): Number of QIDs to process per batch.

    Returns:
        dict: Dictionary of QID → name/aliases data.
    """
    # Load existing data if available to avoid re-processing
    data = utils.load_json(output_path)
    total = len(qids)

    for i in range(0, total, batch_size):
        batch = qids[i:i + batch_size]
        data = process_batch(batch, data)

        # Save progress to disk after each batch
        utils.save_json(output_path, data)
        print(f"Batch {i // batch_size + 1} saved ({len(data)}/{len(qids)})")

    return data


if __name__ == "__main__":
    # Example
    result = get_name_and_aliases("Q6279")
    print(result)
    print(get_all_us_legislator_qids())
