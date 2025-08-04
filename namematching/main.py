import sys
import numpy as np
import pickle
import tensorflow as tf
from namematching.metrics.extract_features import extract_individual_features
from namematching.model.prepare_data import char_tokenizer
from config import CONFIG

with open(CONFIG.export_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = tf.keras.models.load_model(
    CONFIG.export_dir / "namematching_model.keras"
    )


def main(name1, name2):
    x1 = np.expand_dims(char_tokenizer(name1), axis=0)
    x2 = np.expand_dims(char_tokenizer(name2), axis=0)
    features = extract_individual_features(name1, name2)
    features_scaled = scaler.transform(features)
    score = model.predict([x1, x2, features_scaled])[0][0]
    print(f"Similarité entre '{name1}' et '{name2}': {score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
