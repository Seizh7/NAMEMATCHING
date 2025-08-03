from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Concatenate,
)
from tensorflow.keras.models import Model


def encode_branch(input_layer, embedding_layer, conv_layer):
    """
    Sub-network for encoding a name input.

    Applies: Embedding → Conv1D → GlobalMaxPooling.

    Args:
        input_layer (tf.Tensor): Input tensor for a character sequence.
        embedding_layer (tf.keras.layers.Embedding): Shared embedding layer.
        conv_layer (tf.keras.layers.Conv1D): Shared convolutional layer.

    Returns:
        tf.Tensor: Encoded representation of the input sequence.
    """
    x = embedding_layer(input_layer)
    x = conv_layer(x)
    x = GlobalMaxPooling1D()(x)
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
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    return output


def build_namematching_model(
    max_len, char_vocab_size, embed_dim, num_features
):
    """
    Builds a neural model for name matching using character-level embeddings
    and handcrafted features.

    Args:
        max_len (int): Maximum length of character sequences.
        char_vocab_size (int): Size of the character vocabulary.
        embed_dim (int): Dimension of the embedding vector.
        num_features (int): Number of additional handcrafted features.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    # Inputs: two character sequences + classical features
    input1 = Input(shape=(max_len,))
    input2 = Input(shape=(max_len,))
    input_feats = Input(shape=(num_features,))

    # Shared embedding and convolution layers for both name inputs
    embedding = Embedding(
        input_dim=char_vocab_size,
        output_dim=embed_dim,
        input_length=max_len
    )
    conv = Conv1D(filters=64, kernel_size=3, activation='relu')

    # Encode both name1 and name2
    x1 = encode_branch(input1, embedding, conv)
    x2 = encode_branch(input2, embedding, conv)

    # Concatenate both encoded names and the additional features
    merged = Concatenate()([x1, x2, input_feats])
    output = build_classifier(merged)

    # Build and compile the model
    model = Model(inputs=[input1, input2, input_feats], outputs=output)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
