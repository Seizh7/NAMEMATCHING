from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Concatenate,
    Dropout,
)
from tensorflow.keras.models import Model


def encode_branch(input_layer, embedding_layer, bilstm1, bilstm2):
    """
    Encode a name sequence using a shared embedding + stacked BiLSTMs.

    Args:
        input_layer (tf.Tensor): Integer index sequence (batch, max_len).
        embedding_layer (tf.keras.layers.Embedding): Shared embedding.

    Returns:
        tf.Tensor: Fixed-size contextual encoding vector.
    """
    x = embedding_layer(input_layer)  # (batch, T, embed_dim) with masking
    # Apply shared BiLSTM stack (weight sharing across both name branches)
    x = bilstm1(x)
    x = bilstm2(x)
    return x


def build_classifier(concatenated_input):
    """
    Builds dense layers for classification after feature fusion.

    Args:
        concatenated_input (tf.Tensor): Concatenated representation
        of both names and extra features.

    Returns:
        tf.Tensor: Output layer with sigmoid activation.
    """
    x = Dense(64, activation='relu')(concatenated_input)
    x = Dropout(0.5)(x)  # Drop 50% of activations during training
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)  # Lower dropout on the smaller layer
    output = Dense(1, activation='sigmoid')(x)
    return output


def build_namematching_model(
    max_len,
    char_vocab_size,
    embed_dim,
    num_features,
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
    input1 = Input(shape=(max_len,), name="name1_indices")
    input2 = Input(shape=(max_len,), name="name2_indices")
    input_feats = Input(shape=(num_features,), name="extra_features")

    # Shared embedding for both inputs; mask_zero ignores padding index 0
    embedding = Embedding(
        input_dim=char_vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        name="char_embedding",
    )  # input_length deprecated

    # Shared BiLSTM encoder layers (weights reused for both names)
    bilstm1 = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_1")
    bilstm2 = Bidirectional(LSTM(32, return_sequences=False), name="bilstm_2")

    # Encode both names with the same BiLSTM encoder branch
    x1 = encode_branch(input1, embedding, bilstm1, bilstm2)
    x2 = encode_branch(input2, embedding, bilstm1, bilstm2)

    # Fuse encoded names with handcrafted features
    merged = Concatenate(name="fusion")([x1, x2, input_feats])

    # Classification head (defined elsewhere)
    output = build_classifier(merged)

    model = Model(
        inputs=[input1, input2, input_feats],
        outputs=output,
        name="NameMatchingBiLSTM",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
