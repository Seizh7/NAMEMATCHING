import scraper
import json


def main():
    # Get all QIDs of U.S. legislators from Wikidata
    qids = scraper.get_all_us_legislator_qids()

    # Fetch names and aliases for each legislator
    senators = scraper.get_all_names_with_aliases(qids)

    # Save the results to a JSON file
    path = "legislators_aliases.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(senators, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
