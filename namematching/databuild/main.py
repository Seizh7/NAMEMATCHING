import utils
import namematching.databuild.pairs_generator as pairs_generator


def main():
    output_csv = "data/names_pairs.csv"
    data = utils.load_json("data/names_and_aliases.json")

    pairs_generator.generate_and_save_pairs(
        data,
        output_path=output_csv,
        batch_size=50,
        n_negatives=3
    )


if __name__ == "__main__":
    main()
