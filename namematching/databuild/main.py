import utils
import namematching.databuild.pairs_generator as pairs_generator


def main():
    data = utils.load_json("data/names_and_aliases.json")
    output_csv = "data/pairs_generated.csv"

    pairs_generator.generate_and_save_pairs(
        data,
        output_path=output_csv,
        batch_size=50
    )


if __name__ == "__main__":
    main()
