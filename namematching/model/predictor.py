import numpy as np
from namematching.model.prepare_data import char_tokenizer
from namematching.metrics.extract_features import extract_individual_features


class ModelPredictor:
    """
    Utility class for running predictions with a trained name-matching model.

    This class handles all preprocessing and prediction steps.
    """

    def __init__(self, model, feature_scaler):
        """
        Initialize the predictor.

        Args:
            model (tf.keras.Model): Trained Keras model.
            feature_scaler (sklearn.preprocessing.StandardScaler):
                Scaler used to normalize handcrafted features.
        """
        self.model = model
        self.feature_scaler = feature_scaler

    def predict_similarity(self, name1, name2):
        """
        Predicts similarity score between two names.

        Args:
            name1 (str): First name string.
            name2 (str): Second name string.

        Returns:
            float: Similarity score in [0, 1], higher = more similar.
        """
        # Character-level tokenization (with padding/truncation)
        name1_tokenized = np.expand_dims(char_tokenizer(name1), 0)
        name2_tokenized = np.expand_dims(char_tokenizer(name2), 0)

        # Extract handcrafted similarity features
        features = extract_individual_features(name1, name2)

        # Scale features with training-time StandardScaler
        features_scaled = self.feature_scaler.transform(features)

        # Run model inference
        similarity_score = self.model.predict(
            [name1_tokenized, name2_tokenized, features_scaled],
            verbose=0
        )[0][0]

        return similarity_score

    def evaluate_pairs(self, name_pairs, print_results=True):
        """
        Evaluate a list of name pairs.

        Args:
            name_pairs (list[tuple[str, str]]): List of (name1, name2) pairs.
            print_results (bool): If True, prints similarity scores.

        Returns:
            list[tuple[str, str, float]]: Each element is
                (name1, name2, similarity_score).
        """
        results = []

        if print_results:
            print("\n=== Evaluating name pairs ===")

        for name1, name2 in name_pairs:
            similarity_score = self.predict_similarity(name1, name2)
            results.append((name1, name2, similarity_score))

            if print_results:
                print(f"Similarity '{name1}' <> '{name2}' :")
                print(f"\t - {similarity_score:.4f}")

        return results
