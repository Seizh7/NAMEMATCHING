# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pickle
import tensorflow as tf
from namematching.metrics.extract_features import extract_individual_features
from namematching.model.predictor import char_tokenizer
from namematching.resources import get_model_path, get_scaler_path


class NameMatcher:
    """
    Wrapper class around a trained name-matching model.
    Provides utilities to compute similarity scores, binary matches,
    and batch evaluation.
    """

    def __init__(self):
        """
        Initialize the matcher with a pre-trained model and scaler.

        Args:
            model_path : Path to the saved Keras model.
            scaler_path : Path to the saved sklearn scaler.
        """
        # Load trained Keras model
        model_path = get_model_path()
        self.model = tf.keras.models.load_model(
            model_path,
            safe_mode=False
        )

        # Load fitted feature scaler
        scaler_path = get_scaler_path()
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def similarity(self, name1, name2):
        """
        Compute similarity score between two names.

        Args:
            name1 : First name string.
            name2 : Second name string.

        Returns:
            float: Similarity score in [0, 1].
        """
        # Encode character-level inputs
        name1_char_indices = np.expand_dims(char_tokenizer(name1), axis=0)
        name2_char_indices = np.expand_dims(char_tokenizer(name2), axis=0)

        # Extract and scale handcrafted features
        features = extract_individual_features(name1, name2)
        features_scaled = self.scaler.transform(features)

        # Run model prediction
        score = self.model.predict(
            [name1_char_indices, name2_char_indices, features_scaled],
            verbose=0
        )[0][0]
        return float(score)

    def is_match(self, name1, name2, threshold: float = 0.5):
        """
        Decide whether two names are considered a match.

        Args:
            name1 : First name string.
            name2 : Second name string.
            threshold (float): Decision threshold on similarity score.

        Returns:
            bool: True if similarity >= threshold, else False.
        """
        return self.similarity(name1, name2) >= threshold

    def batch_similarity(self, name_pairs):
        """
        Compute similarity scores for a list of name pairs.

        Args:
            name_pairs : List of (name1, name2) pairs.

        Returns:
            list[float]: List of similarity scores.
        """
        return [self.similarity(n1, n2) for n1, n2 in name_pairs]
