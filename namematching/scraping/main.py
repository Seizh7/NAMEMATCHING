import namematching.scraping.scraper as scraper
import os


def main():
    print("Starting scraping of names and aliases from Wikidata.")

    # Retrieve all Wikidata QIDs for U.S. legislators (Senate + House)
    qids = scraper.get_all_us_legislator_qids()
    print(f"Total QIDs to process: {len(qids)}")

    # Define output path and ensure directory exists
    output_path = "data/names_and_aliases.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process the QIDs in batches of 100 and save progress after each batch
    scraper.get_all_names_with_aliases(qids, output_path, batch_size=50)

    print("Scraping complete.")


if __name__ == "__main__":
    main()
