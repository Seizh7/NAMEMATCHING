from namematching.metrics.extract_features import extract_features


def main():
    extract_features(
        input_path="data/pairs_generated.csv",
        output_path="data/pairs_and_features.csv",
        batch_size=5000
    )


if __name__ == "__main__":
    main()
