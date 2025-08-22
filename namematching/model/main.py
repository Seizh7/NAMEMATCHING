import os
import pickle
from sklearn.metrics import classification_report
from namematching.model.prepare_data import load_and_prepare_data
from namematching.model.build_model import build_namematching_model
from config import CONFIG
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(CONFIG.export_dir, "namematching_model.keras")
TOKENIZER_PATH = os.path.join(CONFIG.export_dir, "char_tokenizer.pkl")
SCALER_PATH = os.path.join(CONFIG.export_dir, "scaler.pkl")
MAX_LEN = 40
EMBED_DIM = 32


def main():
    # Create model directory if needed
    os.makedirs(CONFIG.export_dir, exist_ok=True)

    # Load and prepare data
    (
        (train_name1_sequences, train_name2_sequences,
         train_features, train_labels),
        (test_name1_sequences, test_name2_sequences,
         test_features, test_labels),
        feature_scaler,
        char_to_index_mapping,
    ) = load_and_prepare_data(CONFIG.data_dir / "pairs_and_features.csv")

    # Build the model
    model = build_namematching_model(
        max_len=MAX_LEN,
        char_vocab_size=len(char_to_index_mapping) + 1,
        embed_dim=EMBED_DIM,
        num_features=train_features.shape[1],
    )

    # Train the model
    training_history = model.fit(
        [train_name1_sequences, train_name2_sequences, train_features],
        train_labels,
        validation_data=([test_name1_sequences, test_name2_sequences,
                         test_features], test_labels),
        epochs=5,
        batch_size=32,
    )

    # Visualization of the training curves
    plt.plot(training_history.history['accuracy'], label='Train Accuracy')
    plt.plot(training_history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.export_dir, "training_accuracy.png"))
    plt.close()

    # Save the model and preprocessing tools
    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(char_to_index_mapping, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(feature_scaler, f)

    # Evaluate on test set
    binary_predictions = (
        model.predict([test_name1_sequences, test_name2_sequences,
                      test_features]) > 0.5
    ).astype(int)

    print(classification_report(test_labels, binary_predictions))


if __name__ == "__main__":
    main()
