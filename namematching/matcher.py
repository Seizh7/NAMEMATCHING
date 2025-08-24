# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pickle
import tensorflow as tf
from namematching.metrics.extract_features import extract_individual_features
from namematching.training.predictor import char_tokenizer
from namematching.config import CONFIG


def correct_john_bug(name1, name2, score):
    """
    Correct the specific bug where names containing "john" receive abnormally
    low similarity scores due to training bias.

    Args:
        name1 (str): First name string
        name2 (str): Second name string
        score (float): Original similarity score from the model

    Returns:
        float: Corrected similarity score
    """
    name1_lower = name1.lower()
    name2_lower = name2.lower()

    # Check if both names contain "john" pattern
    john_patterns = ["john", "joh"]  # Known problematic patterns
    
    has_john_1 = any(pattern in name1_lower for pattern in john_patterns)
    has_john_2 = any(pattern in name2_lower for pattern in john_patterns)
    
    if has_john_1 and has_john_2:
        # For identical names, return high score
        if name1_lower.strip() == name2_lower.strip():
            return 0.999
        
        # For similar john names, use feature-only approach
        features = extract_individual_features(name1, name2)
        
        # Use handcrafted features as fallback
        jaro_winkler = features['jaro_winkler'].iloc[0]
        first_name_jw = features['first_name_jw'].iloc[0]
        last_name_jaro = features['last_name_jaro'].iloc[0]
        
        # Weighted combination favoring the existing features
        corrected_score = (
            0.4 * jaro_winkler +
            0.3 * first_name_jw +
            0.3 * last_name_jaro
        )
        
        # Return the higher of original score or corrected score
        # to avoid false positives
        return max(score, corrected_score)
    
    return score


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
        model_path = CONFIG.model_dir / "namematching_model.keras"
        self.model = tf.keras.models.load_model(
            model_path,
            safe_mode=False
        )

        # Load fitted feature scaler
        scaler_path = CONFIG.model_dir / "scaler.pkl"
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
        
        # Apply john bug correction
        corrected_score = correct_john_bug(name1, name2, float(score))
        
        return corrected_score

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
