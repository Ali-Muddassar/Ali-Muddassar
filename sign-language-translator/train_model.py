"""
Step 2 — Feature Extraction + Model Training
=============================================
Reads hand-sign images from data/, extracts MediaPipe landmark features,
trains a dense neural network, and saves the model.

Usage:
    python train_model.py

Output:
    model/sign_language_model.h5
    model/label_map.npy
"""

import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # type: ignore

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "label_map.npy")

EPOCHS     = 30
BATCH_SIZE = 32
TEST_SPLIT = 0.2
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands


def extract_landmarks(image_path: str, hands) -> np.ndarray | None:
    """Return a flat (63,) array of x,y,z for all 21 landmarks, or None."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lm   = results.multi_hand_landmarks[0].landmark
    data = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
    return data


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    classes = sorted(os.listdir(DATA_DIR))

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    ) as hands:
        for label in tqdm(classes, desc="Extracting landmarks"):
            class_dir = os.path.join(DATA_DIR, label)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                features = extract_landmarks(
                    os.path.join(class_dir, img_file), hands
                )
                if features is not None:
                    X.append(features)
                    y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(input_dim: int, num_classes: int) -> models.Sequential:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="sign_language_classifier",
    )
    return model


def plot_history(history, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Training plot saved → {save_path}")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    print("[INFO] Loading dataset …")
    X, y_raw = load_dataset()
    if len(X) == 0:
        raise RuntimeError(
            "No landmark data found. Run data_collection.py first."
        )
    print(f"[INFO] Samples: {len(X)}, Features: {X.shape[1]}")

    # 2. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    np.save(LABEL_PATH, le.classes_)
    print(f"[INFO] Classes ({len(le.classes_)}): {list(le.classes_)}")

    # 3. Train / test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    # 4. Build & compile model
    model = build_model(X.shape[1], len(le.classes_))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 5. Train
    cb_list = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    # 6. Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[INFO] Validation accuracy: {acc * 100:.2f}%")

    # 7. Save
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")

    # 8. Plot
    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))
    print("\n[INFO] Training complete!  Run app.py to start the translator.")


if __name__ == "__main__":
    train()
