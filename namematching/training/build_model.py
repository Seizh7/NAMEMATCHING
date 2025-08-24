# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Concatenate,
    Dropout,
    Lambda,
    Multiply,
    Subtract,
    Add,
    ReLU
)
from tensorflow.keras.models import Model


def encode_branch(input_layer, embedding_layer, first_bilstm, second_bilstm):
    """
    Encode a name sequence using a shared embedding + stacked BiLSTMs.

    Args:
        input_layer (tf.Tensor): Integer index sequence (batch, max_len).
        embedding_layer (tf.keras.layers.Embedding): Shared embedding.
        first_bilstm (tf.keras.layers.Bidirectional): First BiLSTM layer.
        second_bilstm (tf.keras.layers.Bidirectional): Second BiLSTM layer.

    Returns:
        tf.Tensor: Fixed-size contextual encoding vector.
    """
    embedded_sequence = embedding_layer(input_layer)
    first_bilstm_output = first_bilstm(embedded_sequence)
    final_encoding = second_bilstm(first_bilstm_output)
    return final_encoding


def build_classifier(fused_representation, feature_input):
    """
    Classification head with learned homonym penalty branch.

    Args:
        fused_representation (tf.Tensor): Fused representation from both
            name branches.
        feature_input (tf.Tensor): Input tensor for additional features.

    Returns:
        tf.Tensor: Output tensor for classification.
    """
    hidden_layer = Dense(48, activation='relu')(fused_representation)
    hidden_layer = Dropout(0.5)(hidden_layer)
    hidden_layer = Dense(24, activation='relu')(hidden_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)
    base_logits = Dense(
        1, activation='linear', name='base_logits'
    )(hidden_layer)

    # Penalty branch
    penalty_hidden = Dense(16, activation='relu')(feature_input)
    penalty_hidden = Dropout(0.2)(penalty_hidden)
    penalty = Dense(
        1, activation='relu', name='homonym_penalty'
    )(penalty_hidden)

    adjusted_logits = Subtract(name='logits_adjusted')([base_logits, penalty])
    return adjusted_logits  # Return logits instead of sigmoid


def create_penalties(
        features_scaled,
        idx_first_name_jaro, idx_last_name_jaro,
        thresh_first_under_same_last,
        thresh_last_under_same_first
):
    """
    Create penalty terms from scaled features.

    Args:
        features_scaled (tf.Tensor): Scaled feature tensor.
        idx_first_name_jaro (int): Index for first name Jaro similarity.
        idx_last_name_jaro (int): Index for last name Jaro similarity.
        thresh_first_under_same_last (float): Threshold for first name
            under same last.
        thresh_last_under_same_first (float): Threshold for last name
            under same first.

    Returns:
        tuple: A tuple containing the extracted similarities and penalty terms.

    """
    # Extract similarities
    first_sim = Lambda(
        lambda f: f[:, idx_first_name_jaro:idx_first_name_jaro + 1],
        name="first_name_sim"
    )(features_scaled)
    last_sim = Lambda(
        lambda f: f[:, idx_last_name_jaro:idx_last_name_jaro + 1],
        name="last_name_sim"
    )(features_scaled)

    # Create penalty terms
    threshold_first = tf.constant(
        thresh_first_under_same_last,
        dtype=tf.float32
    )
    threshold_last = tf.constant(
        thresh_last_under_same_first,
        dtype=tf.float32
    )

    pen_last_low_first = Multiply(name="pen_sameLast_diffFirst")(
        [last_sim, ReLU()(Lambda(lambda s: threshold_first - s)(first_sim))]
    )
    pen_first_low_last = Multiply(name="pen_sameFirst_diffLast")(
        [first_sim, ReLU()(Lambda(lambda s: threshold_last - s)(last_sim))]
    )

    return first_sim, last_sim, pen_last_low_first, pen_first_low_last


def apply_final_transformations(
    logits, first_sim, last_sim, pen1, pen2,
    alpha_same_last_diff_first, alpha_same_first_diff_last
):
    """
    Apply penalties, gating, and reinforcements to get final probability.

    Args:
        logits (tf.Tensor): Logits from the model.
        first_sim (tf.Tensor): First name similarity.
        last_sim (tf.Tensor): Last name similarity.
        pen1 (tf.Tensor): Penalty term 1.
        pen2 (tf.Tensor): Penalty term 2.
        alpha_same_last_diff_first (float): Weight for same last name,
            different first name penalty.
        alpha_same_first_diff_last (float): Weight for same first name,
            different last name penalty.

    Returns:
        tf.Tensor: Final probability after applying all transformations.
    """

    # Apply penalties in logit space
    adjusted_logits = Lambda(
        lambda args: (
            args[0]
            - alpha_same_last_diff_first * args[1]
            - alpha_same_first_diff_last * args[2]
        ),
        name="penalty_adjustment"
    )([logits, pen1, pen2])

    # Convert to probability
    prob = tf.keras.activations.sigmoid(adjusted_logits)

    # Apply gating and reinforcements
    first_square = Multiply()([first_sim, first_sim])
    last_square = Multiply()([last_sim, last_sim])
    last_cube = Multiply()([last_square, last_sim])

    # Combined gate with geometric mean and strict thresholds
    combined_gate = Lambda(
        lambda t: tf.where(
            tf.logical_or(t[0] >= 0.6, t[1] >= 0.6),
            tf.sqrt(t[0] * t[1]),
            tf.constant(0.05, dtype=t[0].dtype)
        )
    )([first_sim, last_sim])

    # Add minimum similarity threshold to prevent high scores for completely
    # different names
    min_similarity_gate = Lambda(
        lambda t: tf.where(
            tf.logical_or(
                # Both names have some similarity
                tf.logical_and(t[0] >= 0.3, t[1] >= 0.3),
                # High first name similarity
                tf.logical_and(t[0] >= 0.7, t[1] >= 0.1),
                # High last name similarity
                tf.logical_and(t[0] >= 0.1, t[1] >= 0.7)
            ),
            tf.constant(1.0, dtype=t[0].dtype),
            # Severely penalize low similarities
            tf.maximum(t[0] * t[1] * 2.0, tf.constant(0.01, dtype=t[0].dtype))
        )
    )([first_sim, last_sim])

    # Apply all transformations including the new minimum similarity gate
    final_prob = Multiply(name="output")(
        [
            prob,
            first_sim,
            last_sim,
            first_square,
            last_cube,
            combined_gate,
            min_similarity_gate
        ]
    )

    return final_prob


def build_namematching_model(
    max_len,
    char_vocab_size,
    embed_dim,
    num_features,
    scaler_mean=None,
    scaler_scale=None,
    idx_first_name_jaro=0,
    idx_last_name_jaro=1,
    alpha_same_last_diff_first=6.0,
    alpha_same_first_diff_last=5.0,
    thresh_first_under_same_last=0.8,
    thresh_last_under_same_first=0.85,
):
    """
    Builds a neural model for name matching using character-level embeddings
    encoded by a shared stacked BiLSTM for each name plus handcrafted features.

    Args:
        max_len (int): Maximum length of character sequences.
        char_vocab_size (int): Size of the character vocabulary.
        embed_dim (int): Dimension of the embedding vector.
        num_features (int): Number of additional handcrafted features.
        scaler_mean (np.ndarray, optional): Mean values for feature scaling.
        scaler_scale (np.ndarray, optional): Scale values for feature scaling.
        idx_first_name_jaro (int, optional): Index for first name
            Jaro similarity.
        idx_last_name_jaro (int, optional): Index for last name
            Jaro similarity.
        alpha_same_last_diff_first (float, optional): Penalty weight for same
            last name, different first name.
        alpha_same_first_diff_last (float, optional): Penalty weight for same
            first name, different last name.
        thresh_first_under_same_last (float, optional): Threshold for first
            name under same last name.
        thresh_last_under_same_first (float, optional): Threshold for last
            name under same first name.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    # Inputs
    name1_input = Input(shape=(max_len,), name="name1_indices")
    name2_input = Input(shape=(max_len,), name="name2_indices")
    features_input = Input(shape=(num_features,), name="extra_features")

    # Shared embedding and BiLSTM layers
    embedding = Embedding(
        input_dim=char_vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        name="char_embedding"
    )
    first_bilstm = Bidirectional(
        LSTM(64, return_sequences=True),
        name="bilstm_1"
    )
    second_bilstm = Bidirectional(
        LSTM(32, return_sequences=False),
        name="bilstm_2"
    )

    # Encode both names
    name1_encoded = encode_branch(
        name1_input, embedding, first_bilstm, second_bilstm
    )
    name2_encoded = encode_branch(
        name2_input, embedding, first_bilstm, second_bilstm
    )

    # Create interaction features
    difference = Subtract()([name1_encoded, name2_encoded])
    absolute_difference = Add()([
        ReLU()(difference),
        ReLU()(Lambda(lambda t: -t)(difference))
    ])
    elem_prod = Multiply()([name1_encoded, name2_encoded])

    # Fuse all features
    fused = Concatenate(name="fusion")([
        name1_encoded,
        name2_encoded,
        absolute_difference,
        elem_prod,
        features_input
    ])

    # Get base logits from classifier
    logits = build_classifier(fused, features_input)

    # Apply penalties and transformations if scaler is provided
    if scaler_mean is not None and scaler_scale is not None:
        # Restore original scale
        features_scaled = Lambda(
            lambda f: (
                f * tf.constant(scaler_scale, dtype=tf.float32)
                + tf.constant(scaler_mean, dtype=tf.float32)
            ),
            name="orig_features"
        )(features_input)

        # Create penalties
        first_sim, last_sim, pen1, pen2 = create_penalties(
            features_scaled, idx_first_name_jaro, idx_last_name_jaro,
            thresh_first_under_same_last, thresh_last_under_same_first
        )

        # Apply final transformations
        output = apply_final_transformations(
            logits, first_sim, last_sim, pen1, pen2,
            alpha_same_last_diff_first, alpha_same_first_diff_last
        )
    else:
        output = tf.keras.activations.sigmoid(logits)

    model = Model(
        inputs=[name1_input, name2_input, features_input],
        outputs=output,
        name="NameMatchingBiLSTM"
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    return model
