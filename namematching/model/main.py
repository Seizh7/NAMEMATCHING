import os
import pickle
from sklearn.metrics import classification_report
from namematching.model.prepare_data import load_and_prepare_data
from namematching.model.build_model import build_namematching_model
from config import CONFIG

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
        (train_X1, train_X2, train_feats, y_train),
        (test_X1, test_X2, test_feats, y_test),
        scaler,
        char_to_idx,
    ) = load_and_prepare_data(CONFIG.data_dir / "pairs_and_features.csv")

    # Build the model
    model = build_namematching_model(
        max_len=MAX_LEN,
        char_vocab_size=len(char_to_idx) + 1,
        embed_dim=EMBED_DIM,
        num_features=train_feats.shape[1],
    )

    # Train the model
    model.fit(
        [train_X1, train_X2, train_feats],
        y_train,
        validation_data=([test_X1, test_X2, test_feats], y_test),
        epochs=15,
        batch_size=32,
    )

    # Save the model and preprocessing tools
    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(char_to_idx, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate on test set
    predictions = (
        model.predict([test_X1, test_X2, test_feats]) > 0.5
    ).astype(int)

    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
