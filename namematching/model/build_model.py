import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Concatenate,
    Dropout,
    Activation,
    Lambda,
    Multiply,
    Subtract,
)
from tensorflow.keras.models import Model


def encode_branch(input_layer, embedding_layer, first_bilstm, second_bilstm):
    """
    Encode a name sequence using a shared embedding + stacked BiLSTMs.

    Args:
        input_layer (tf.Tensor): Integer index sequence (batch, max_len).
        embedding_layer (tf.keras.layers.Embedding): Shared embedding.

    Returns:
        tf.Tensor: Fixed-size contextual encoding vector.
    """
    embedded_sequence = embedding_layer(input_layer)
    # Apply shared BiLSTM stack (weight sharing across both name branches)
    first_bilstm_output = first_bilstm(embedded_sequence)
    final_encoding = second_bilstm(first_bilstm_output)
    return final_encoding


def build_classifier(fused_representation, feature_input):
    """Classification head with learned homonym penalty branch.

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
        1,
        activation='linear',
        name='base_logits'
    )(hidden_layer)

    # Penalty branch — encourage positive value when should decrease
    penalty_hidden = Dense(16, activation='relu')(feature_input)
    penalty_hidden = Dropout(0.2)(penalty_hidden)
    penalty = Dense(
        1, activation='relu',
        name='homonym_penalty'
    )(penalty_hidden)

    adjusted_logits = Subtract(name='logits_adjusted')([base_logits, penalty])
    output = Activation('sigmoid', name='output')(adjusted_logits)
    return output


def adjust_with_penalties(
    penalty_args,
    alpha_same_last_diff_first,
    alpha_same_first_diff_last
):
    """Adjust the output probability by applying penalties in logit space.

    Args:
        penalty_args: Tuple of (prob, penalty1, penalty2)
        alpha_same_last_diff_first: Penalty weight for same last name but
                                    different first name
        alpha_same_first_diff_last: Penalty weight for same first name but
                                    different last name

    Returns:
        Adjusted probability after applying penalties
    """
    probability, penalty_same_last, penalty_same_first = penalty_args
    epsilon = 1e-6
    logit = (tf.math.log(probability + epsilon) -
             tf.math.log(1. - probability + epsilon))
    logit -= alpha_same_last_diff_first * penalty_same_last
    logit -= alpha_same_first_diff_last * penalty_same_first
    return tf.math.sigmoid(logit)


def gate_output(gating_args):
    """
    Gating mechanism to adjust the final probability when either first or last

    Args:
        gating_args: Tuple of (prob, first_sim, last_sim)

    Returns:
        Gated probability
    """
    probability, first_similarity, last_similarity = gating_args
    gate = first_similarity * last_similarity
    return probability * gate


def build_namematching_model(
    max_len,
    char_vocab_size,
    embed_dim,
    num_features,
    scaler_mean=None,
    scaler_scale=None,
    idx_first_name_jaro=0,
    idx_last_name_jaro=3,
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

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    # Inputs: two character-index sequences + feature vector
    name1_input = Input(shape=(max_len,), name="name1_indices")
    name2_input = Input(shape=(max_len,), name="name2_indices")
    features_input = Input(shape=(num_features,), name="extra_features")

    # Shared embedding for both inputs; mask_zero ignores padding index 0
    embedding = Embedding(
        input_dim=char_vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        name="char_embedding",
    )

    # Shared BiLSTM encoder layers (weights reused for both names)
    first_bilstm = Bidirectional(LSTM(64, return_sequences=True),
                                 name="bilstm_1")
    second_bilstm = Bidirectional(LSTM(32, return_sequences=False),
                                  name="bilstm_2")

    # Encode both names with the same BiLSTM encoder branch
    name1_encoded = encode_branch(name1_input, embedding,
                                  first_bilstm, second_bilstm)
    name2_encoded = encode_branch(name2_input, embedding,
                                  first_bilstm, second_bilstm)

    # Interaction features to help the network model similarity explicitly
    absolute_difference = Lambda(
        lambda t: tf.math.abs(t[0] - t[1]),
        name='abs_diff'
    )([name1_encoded, name2_encoded])

    element_product = Multiply(
        name='elem_prod'
    )([name1_encoded, name2_encoded])

    # Fuse encoded names with handcrafted features
    merged_features = Concatenate(name="fusion")([
        name1_encoded, name2_encoded, absolute_difference,
        element_product, features_input
    ])

    output = build_classifier(merged_features, features_input)

    # Deterministic penalties applied in logit space.
    if scaler_mean is not None and scaler_scale is not None:
        scaler_mean_tensor = tf.constant(scaler_mean, dtype=tf.float32)
        scaler_scale_tensor = tf.constant(scaler_scale, dtype=tf.float32)
        original_features = Lambda(
            lambda f: f * scaler_scale_tensor + scaler_mean_tensor,
            name='orig_features'
        )(features_input)
        first_name_similarity = Lambda(
            lambda f: tf.gather(f, [idx_first_name_jaro], axis=1),
            name='first_name_sim'
        )(original_features)
        last_name_similarity = Lambda(
            lambda f: tf.gather(f, [idx_last_name_jaro], axis=1),
            name='last_name_sim'
        )(original_features)
        # same last + low first sim: last_sim * relu(thresh - first_sim)
        penalty_same_last_diff_first = Lambda(
            lambda t: t[0] * tf.nn.relu(thresh_first_under_same_last - t[1]),
            name='pen_sameLast_diffFirst'
        )([last_name_similarity, first_name_similarity])
        # same first + low last sim: first_sim * relu(thresh - last_sim)
        penalty_same_first_diff_last = Lambda(
            lambda t: t[0] * tf.nn.relu(thresh_last_under_same_first - t[1]),
            name='pen_sameFirst_diffLast'
        )([first_name_similarity, last_name_similarity])

        # Adjust output with penalties in logit space
        output = Lambda(
            lambda args: adjust_with_penalties(
                args, alpha_same_last_diff_first, alpha_same_first_diff_last
            ),
            name='output_with_penalties'
        )([output, penalty_same_last_diff_first, penalty_same_first_diff_last])

        # Gating: damp final prob when either first or last similarity low
        output = Lambda(
            gate_output,
            name='output_gated'
        )([output, first_name_similarity, last_name_similarity])

    model = Model(inputs=[name1_input, name2_input, features_input],
                  outputs=output,
                  name="NameMatchingBiLSTM")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
